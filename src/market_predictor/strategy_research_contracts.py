from __future__ import annotations

import hashlib
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_training import load_swing_training_config
from market_predictor.execution_policy import (
    EXECUTION_POLICY_ID,
    EXECUTION_POLICY_SHA256,
)
from market_predictor.intraday.contracts import (
    IntradayDatasetConfig,
    IntradayPromotionConfig,
    IntradayTrainingConfig,
)
from market_predictor.strategy_governance import (
    CATALOG_ID_PATTERN,
    StrategyExecutionLedger,
    validate_strategy_execution_ledger,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.core.errors import ArtifactIntegrityError, DataReadinessError

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REQUIRED_CONTRACT_BINDINGS = frozenset(
    {
        "swing_dataset",
        "edge_rebuild_strategy_contract",
        "edge_rebuild_swing_training",
        "intraday_dataset",
        "intraday_training",
        "intraday_promotion",
    }
)
REQUIRED_VALIDATION_SCOPES = frozenset(
    {"purged_walk_forward", "unseen_ticker_holdout"}
)
REQUIRED_RETIREMENT_TRIGGERS = frozenset(
    {
        "development_budget_exhausted_without_gate_pass",
        "frozen_hypothesis_fails_both_validation_scopes",
        "required_data_contract_invalidated",
        "shadow_attempt_consumed_without_promotion",
        "semantic_change_requires_new_strategy_version",
    }
)
HypothesisState = Literal[
    "planned",
    "reference_rejected",
    "data_blocked",
    "deferred",
    "candidate_rejected",
    "candidate_passed",
    "promoted",
]


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FileBinding(FrozenContract):
    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def normalized_relative_path(cls, value: str) -> str:
        return _validated_relative_path(value)


class ExecutionPolicyBinding(FrozenContract):
    policy_id: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=SHA256_PATTERN)


class StrategyResearchGovernance(FrozenContract):
    schema_version: Literal["market_predictor.strategy_research_governance.v1"]
    maximum_development_experiments_per_strategy_version: int = Field(
        ge=1,
        le=100,
    )
    maximum_estimator_families_per_strategy_version: int = Field(ge=1, le=10)
    maximum_feature_profiles_per_strategy_version: int = Field(ge=1, le=10)
    maximum_selection_policies_per_strategy_version: int = Field(ge=1, le=10)
    maximum_shadow_attempts_per_strategy_version: Literal[1]
    required_validation_scopes: tuple[str, ...] = Field(min_length=2)
    allowed_estimator_families: tuple[str, ...] = Field(min_length=1)
    allowed_feature_profiles: tuple[str, ...] = Field(min_length=1)
    allowed_selection_policies: tuple[str, ...] = Field(min_length=1)
    retirement_triggers: tuple[str, ...] = Field(min_length=1)
    minimum_label_round_trip_cost_bps: float = Field(ge=0, le=500)
    maximum_process_memory_gib: float = Field(gt=0, le=5)
    swing_purge_rule: Literal["label_horizon_sessions"]
    minimum_intraday_embargo_sessions: int = Field(ge=1, le=10)
    contract_bindings: dict[str, FileBinding]
    execution_policy: ExecutionPolicyBinding

    @model_validator(mode="after")
    def validate_governance(self) -> Self:
        if set(self.contract_bindings) != REQUIRED_CONTRACT_BINDINGS:
            raise ValueError(
                "research policy must bind the exact swing and intraday contracts"
            )
        _require_unique(
            list(self.required_validation_scopes),
            "validation scope",
        )
        _require_unique(
            list(self.allowed_estimator_families),
            "estimator family",
        )
        _require_unique(
            list(self.allowed_feature_profiles),
            "feature profile",
        )
        _require_unique(
            list(self.allowed_selection_policies),
            "selection policy",
        )
        _require_unique(list(self.retirement_triggers), "retirement trigger")
        if set(self.required_validation_scopes) != REQUIRED_VALIDATION_SCOPES:
            raise ValueError("both temporal and unseen-ticker validation are required")
        if not REQUIRED_RETIREMENT_TRIGGERS.issubset(self.retirement_triggers):
            raise ValueError("research policy is missing a mandatory retirement trigger")
        if self.maximum_estimator_families_per_strategy_version > len(
            self.allowed_estimator_families
        ):
            raise ValueError("estimator-family limit exceeds the allowed inventory")
        if self.maximum_feature_profiles_per_strategy_version > len(
            self.allowed_feature_profiles
        ):
            raise ValueError("feature-profile limit exceeds the allowed inventory")
        if self.maximum_selection_policies_per_strategy_version > len(
            self.allowed_selection_policies
        ):
            raise ValueError("selection-policy limit exceeds the allowed inventory")
        combination_ceiling = (
            self.maximum_estimator_families_per_strategy_version
            * self.maximum_feature_profiles_per_strategy_version
            * self.maximum_selection_policies_per_strategy_version
        )
        if self.maximum_development_experiments_per_strategy_version > combination_ceiling:
            raise ValueError(
                "experiment budget exceeds the declared comparison dimensions"
            )
        return self


class ResearchHypothesis(FrozenContract):
    hypothesis_id: str = Field(
        pattern=r"^(?:SWING|INTRADAY|RISK|META)(?:\.[A-Z0-9_]+)+\.V[1-9]\d*\.H1$"
    )
    item_id: str = Field(min_length=1, max_length=128)
    claim: str = Field(min_length=20, max_length=1_000)
    eligible_population: str = Field(min_length=20, max_length=1_000)
    primary_outcome: str = Field(min_length=10, max_length=500)
    comparator: str = Field(min_length=10, max_length=500)
    falsified_when: str = Field(min_length=20, max_length=1_000)
    state: HypothesisState

    @field_validator("item_id")
    @classmethod
    def valid_item_id(cls, value: str) -> str:
        if CATALOG_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid strategy or component ID")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.hypothesis_id != f"{self.item_id}.H1":
            raise ValueError("research hypothesis must be H1 for its strategy version")
        return self


class ResearchHypothesisRegistry(FrozenContract):
    schema_version: Literal["market_predictor.strategy_hypothesis_registry.v1"]
    research_policy: FileBinding
    hypotheses: tuple[ResearchHypothesis, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        _require_unique(
            [hypothesis.hypothesis_id for hypothesis in self.hypotheses],
            "hypothesis",
        )
        _require_unique(
            [hypothesis.item_id for hypothesis in self.hypotheses],
            "hypothesis item",
        )
        return self


class ReferenceModel(FrozenContract):
    reference_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    view: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d|b)$")
    family: str = Field(min_length=1, max_length=128)
    status: Literal["reference_rejected", "implementation_only"]
    strategy_id: None = None
    serving_eligible: Literal[False]
    reason: str = Field(min_length=20, max_length=1_000)
    evidence_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_paths")
    @classmethod
    def normalized_evidence_paths(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(_validated_relative_path(value) for value in values)


class ReferenceModelInventory(FrozenContract):
    schema_version: Literal["market_predictor.reference_model_inventory.v1"]
    models: tuple[ReferenceModel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        _require_unique(
            [model.reference_id for model in self.models],
            "reference model",
        )
        if {model.view for model in self.models} != {"swing", "intraday"}:
            raise ValueError("reference inventory must cover swing and intraday")
        return self


def validate_strategy_research_contracts(
    *,
    ledger_path: Path,
    hypothesis_registry_path: Path,
    policy_path: Path,
    reference_inventory_path: Path,
    repository_root: Path,
    verify_git: bool = True,
) -> dict[str, object]:
    root = repository_root.resolve()
    ledger_report = validate_strategy_execution_ledger(
        ledger_path,
        repository_root=root,
        verify_git=verify_git,
    )
    ledger = StrategyExecutionLedger.model_validate_json(
        _read_required(
            _resolve_repository_path(root, ledger_path),
            "strategy execution ledger",
        )
    )

    registry_bytes = _read_required(
        _resolve_repository_path(root, hypothesis_registry_path),
        "strategy hypothesis registry",
    )
    registry = ResearchHypothesisRegistry.model_validate_json(registry_bytes)
    policy_file = _resolve_repository_path(root, policy_path)
    policy_bytes = _read_required(policy_file, "strategy research policy")
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if registry.research_policy.path != _repository_relative(root, policy_file):
        raise ArtifactIntegrityError(
            "hypothesis registry references a different research policy path"
        )
    if registry.research_policy.sha256 != policy_sha256:
        raise ArtifactIntegrityError(
            "hypothesis registry research policy hash mismatch"
        )
    policy = StrategyResearchGovernance.model_validate(
        tomllib.loads(policy_bytes.decode("utf-8"))
    )

    _validate_contract_bindings(root, policy)
    _validate_shared_assumptions(root, policy)
    _validate_hypothesis_alignment(ledger, registry)

    inventory_bytes = _read_required(
        _resolve_repository_path(root, reference_inventory_path),
        "reference model inventory",
    )
    inventory = ReferenceModelInventory.model_validate_json(inventory_bytes)
    for model in inventory.models:
        for path_value in model.evidence_paths:
            _read_required(
                _resolve_repository_path(root, Path(path_value)),
                f"{model.reference_id} evidence",
            )

    hypothesis_counts = Counter(
        hypothesis.state for hypothesis in registry.hypotheses
    )
    return {
        "schema_version": policy.schema_version,
        "valid": True,
        "ledger_valid": ledger_report["valid"],
        "ledger_sha256": hashlib.sha256(
            _read_required(
                _resolve_repository_path(root, ledger_path),
                "strategy execution ledger",
            )
        ).hexdigest(),
        "catalog_count": len(ledger.catalog),
        "hypothesis_count": len(registry.hypotheses),
        "hypothesis_status": dict(sorted(hypothesis_counts.items())),
        "hypothesis_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "research_policy_sha256": policy_sha256,
        "contract_bindings": {
            name: binding.model_dump(mode="json")
            for name, binding in sorted(policy.contract_bindings.items())
        },
        "maximum_development_experiments_per_strategy_version": (
            policy.maximum_development_experiments_per_strategy_version
        ),
        "maximum_shadow_attempts_per_strategy_version": (
            policy.maximum_shadow_attempts_per_strategy_version
        ),
        "validation_scopes": list(policy.required_validation_scopes),
        "execution_policy_id": policy.execution_policy.policy_id,
        "execution_policy_sha256": policy.execution_policy.sha256,
        "reference_model_count": len(inventory.models),
        "reference_models_serving_eligible": 0,
        "reference_model_inventory_sha256": hashlib.sha256(
            inventory_bytes
        ).hexdigest(),
        "reference_model_ids": [
            model.reference_id for model in inventory.models
        ],
    }


def _validate_contract_bindings(
    root: Path,
    policy: StrategyResearchGovernance,
) -> None:
    for name, binding in policy.contract_bindings.items():
        path = _resolve_repository_path(root, Path(binding.path))
        content = _read_required(path, f"{name} contract")
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != binding.sha256:
            raise ArtifactIntegrityError(f"{name} contract hash mismatch")
    if policy.execution_policy.policy_id != EXECUTION_POLICY_ID:
        raise ArtifactIntegrityError("execution policy ID mismatch")
    if policy.execution_policy.sha256 != EXECUTION_POLICY_SHA256:
        raise ArtifactIntegrityError("execution policy hash mismatch")


def _validate_shared_assumptions(
    root: Path,
    policy: StrategyResearchGovernance,
) -> None:
    swing_dataset = SwingDatasetConfig.model_validate(
        _load_bound_toml(root, policy, "swing_dataset")
    )
    edge_strategy = load_strategy_contract(
        _resolve_repository_path(
            root,
            Path(policy.contract_bindings["edge_rebuild_strategy_contract"].path),
        )
    )
    edge_swing_training = load_swing_training_config(
        _resolve_repository_path(
            root,
            Path(policy.contract_bindings["edge_rebuild_swing_training"].path),
        )
    )
    intraday_dataset = IntradayDatasetConfig.model_validate(
        _load_bound_toml(root, policy, "intraday_dataset")
    )
    intraday_training = IntradayTrainingConfig.model_validate(
        _load_bound_toml(root, policy, "intraday_training")
    )
    IntradayPromotionConfig.model_validate(
        _load_bound_toml(root, policy, "intraday_promotion")
    )

    if min(
        edge_strategy.swing.round_trip_cost_bps,
        intraday_dataset.round_trip_cost_bps,
    ) < policy.minimum_label_round_trip_cost_bps:
        raise DataReadinessError("label-cost floor is below research policy")
    if edge_swing_training.horizon_sessions != edge_strategy.swing.horizon_sessions:
        raise DataReadinessError("swing trainer horizon differs from the edge strategy")
    if edge_swing_training.maximum_trades_per_decision != edge_strategy.swing.maximum_trades_per_decision:
        raise DataReadinessError("swing selection limit differs from the edge strategy")
    if intraday_training.embargo_sessions < policy.minimum_intraday_embargo_sessions:
        raise DataReadinessError("intraday embargo is below research policy")
    if {
        swing_dataset.required_price_feed.lower(),
        intraday_dataset.required_price_feed.lower(),
    } != {"sip"}:
        raise DataReadinessError("strategy research requires SIP feed coverage")
    if {
        swing_dataset.required_adjustment.lower(),
        intraday_dataset.required_adjustment.lower(),
    } != {"all"}:
        raise DataReadinessError("strategy research requires all adjustments")
    if (
        swing_dataset.broad_benchmark.upper()
        != intraday_dataset.broad_benchmark.upper()
        or swing_dataset.growth_benchmark.upper()
        != intraday_dataset.growth_benchmark.upper()
    ):
        raise DataReadinessError("swing and intraday benchmark contracts differ")
    memory_values = (
        swing_dataset.max_build_memory_gb,
        edge_swing_training.maximum_process_memory_gib,
        intraday_dataset.max_build_memory_gb,
        intraday_training.max_training_memory_gb,
    )
    if any(value > policy.maximum_process_memory_gib for value in memory_values):
        raise DataReadinessError("bound contract exceeds the research memory budget")


def _validate_hypothesis_alignment(
    ledger: StrategyExecutionLedger,
    registry: ResearchHypothesisRegistry,
) -> None:
    catalog_by_id = {entry.item_id: entry for entry in ledger.catalog}
    hypothesis_by_id = {
        hypothesis.item_id: hypothesis for hypothesis in registry.hypotheses
    }
    if set(catalog_by_id) != set(hypothesis_by_id):
        raise DataReadinessError(
            "research hypothesis coverage differs from the strategy catalog"
        )
    for item_id, catalog_entry in catalog_by_id.items():
        hypothesis = hypothesis_by_id[item_id]
        if hypothesis.state != catalog_entry.state:
            raise DataReadinessError(
                f"research hypothesis state mismatch: {item_id}"
            )


def _load_bound_toml(
    root: Path,
    policy: StrategyResearchGovernance,
    name: str,
) -> dict[str, object]:
    binding = policy.contract_bindings[name]
    path = _resolve_repository_path(root, Path(binding.path))
    return tomllib.loads(_read_required(path, f"{name} contract").decode("utf-8"))


def _validated_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("repository paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("repository path must be normalized and relative")
    return value


def _resolve_repository_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"artifact escapes repository root: {path}"
        ) from exc
    return resolved


def _repository_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _read_required(path: Path, description: str) -> bytes:
    if not path.is_file():
        raise ArtifactIntegrityError(f"missing {description}: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read {description}: {path}") from exc


def _require_unique(values: list[str], name: str) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate {name} values: {duplicates}")
