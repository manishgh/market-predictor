"""Scoped intraday contract identity and explicit parent-hash migration evidence."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.v3.errors import DataReadinessError

INTRADAY_DATA_CONTRACT_SCHEMA: Final = "edge_rebuild.intraday_data_contract.v1"
INTRADAY_CONTRACT_LINEAGE_SCHEMA: Final = (
    "edge_rebuild.intraday_contract_lineage.v1"
)
DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH: Final = Path(
    "configs/edge_rebuild_intraday_contract_lineage.toml"
)


@dataclass(frozen=True, slots=True)
class IntradayContractIdentity:
    mode: str
    observed_contract_sha256: str
    observed_contract_file_sha256: str | None
    current_contract_sha256: str
    current_contract_file_sha256: str
    intraday_data_contract_sha256: str
    lineage_file_sha256: str | None
    source_commit: str | None


def intraday_data_contract_payload(
    contract: StrategyContract,
) -> dict[str, Any]:
    """Return only fields capable of changing intraday rows or labels."""

    labels = contract.labels
    features = contract.features
    quality = contract.data_quality
    return {
        "schema": INTRADAY_DATA_CONTRACT_SCHEMA,
        "intraday": contract.intraday.model_dump(mode="json"),
        "intraday_universe": contract.intraday_universe.model_dump(mode="json"),
        "methodology": {
            "labeling": contract.methodology.labeling,
            "sampling": contract.methodology.sampling,
        },
        "labels": {
            "benchmark_market": labels.benchmark_market,
            "benchmark_sector_source": labels.benchmark_sector_source,
            "barrier_labels_enabled": labels.barrier_labels_enabled,
            "rank_labels_enabled": labels.rank_labels_enabled,
            "rank_top_quantile": labels.rank_top_quantile,
            "rank_bottom_quantile": labels.rank_bottom_quantile,
            "intraday_rank_within_sector": labels.intraday_rank_within_sector,
            "intraday_minimum_cross_section_for_ranking": (
                labels.intraday_minimum_cross_section_for_ranking
            ),
        },
        "technical_state": {
            "technical_relationship_methods": list(
                features.technical_relationship_methods
            ),
            "rsi_pivot_span_bars": features.rsi_pivot_span_bars,
            "obv_confirmation_lookback_bars": (
                features.obv_confirmation_lookback_bars
            ),
            "efficiency_ratio_lookback_bars": (
                features.efficiency_ratio_lookback_bars
            ),
        },
        "data_quality": {
            "maximum_security_exclusion_fraction": (
                quality.maximum_security_exclusion_fraction
            ),
            "exclusion_unit": quality.exclusion_unit,
            "benchmark_exclusions_allowed": quality.benchmark_exclusions_allowed,
            "market_session_exclusions_allowed": (
                quality.market_session_exclusions_allowed
            ),
        },
    }


def intraday_data_contract_sha256(contract: StrategyContract) -> str:
    payload = json.dumps(
        intraday_data_contract_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_intraday_contract_lineage(
    *,
    observed_contract_sha256: object,
    observed_contract_file_sha256: object | None,
    current_contract: StrategyContract,
    current_contract_path: Path,
    lineage_path: Path = DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
) -> IntradayContractIdentity:
    """Accept an exact current contract or one explicitly scope-equivalent parent."""

    observed = str(observed_contract_sha256 or "").strip()
    observed_file = (
        None
        if observed_contract_file_sha256 is None
        else str(observed_contract_file_sha256).strip()
    )
    current = current_contract.sha256()
    current_file = file_sha256(current_contract_path)
    scoped = intraday_data_contract_sha256(current_contract)
    if observed == current and observed_file in {None, current_file}:
        return IntradayContractIdentity(
            mode="exact_current_contract",
            observed_contract_sha256=observed,
            observed_contract_file_sha256=observed_file,
            current_contract_sha256=current,
            current_contract_file_sha256=current_file,
            intraday_data_contract_sha256=scoped,
            lineage_file_sha256=None,
            source_commit=None,
        )
    lineage = _load_lineage(lineage_path)
    parents = lineage.get("parents")
    if not isinstance(parents, list):
        raise DataReadinessError("intraday contract lineage has no parent records")
    match: Mapping[str, Any] | None = None
    for raw in parents:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("intraday contract lineage parent is malformed")
        if str(raw.get("full_contract_sha256", "")) != observed:
            continue
        expected_file = str(raw.get("contract_file_sha256", ""))
        if observed_file is not None and expected_file != observed_file:
            continue
        if match is not None:
            raise DataReadinessError("intraday contract lineage repeats a parent")
        match = raw
    if match is None:
        raise DataReadinessError(
            "intraday parent contract is neither current nor explicitly equivalent"
        )
    parent_scoped = str(match.get("intraday_data_contract_sha256", ""))
    source_commit = str(match.get("source_commit", "")).strip()
    if not _is_full_git_sha(source_commit):
        raise DataReadinessError(
            "intraday parent contract source commit is not a full immutable Git SHA"
        )
    if (
        parent_scoped != scoped
        or str(lineage.get("current_intraday_data_contract_sha256", "")) != scoped
        or str(match.get("source_path", "")).strip() == ""
        or str(match.get("reason", "")).strip() == ""
    ):
        raise DataReadinessError(
            "intraday parent contract does not match the current scoped contract"
        )
    _verify_parent_snapshot(
        parent=match,
        lineage_path=lineage_path,
        expected_full_contract_sha256=observed,
        expected_file_sha256=str(match.get("contract_file_sha256", "")),
        expected_intraday_sha256=scoped,
    )
    return IntradayContractIdentity(
        mode="verified_scope_equivalent_parent",
        observed_contract_sha256=observed,
        observed_contract_file_sha256=observed_file,
        current_contract_sha256=current,
        current_contract_file_sha256=current_file,
        intraday_data_contract_sha256=scoped,
        lineage_file_sha256=file_sha256(lineage_path),
        source_commit=source_commit,
    )


def _verify_parent_snapshot(
    *,
    parent: Mapping[str, Any],
    lineage_path: Path,
    expected_full_contract_sha256: str,
    expected_file_sha256: str,
    expected_intraday_sha256: str,
) -> None:
    snapshot_reference = str(parent.get("snapshot_path", "")).strip()
    if snapshot_reference == "":
        raise DataReadinessError(
            "intraday parent contract has no immutable snapshot"
        )
    lineage_root = (lineage_path.parent / "lineage").resolve()
    snapshot_path = (lineage_path.parent / snapshot_reference).resolve()
    try:
        snapshot_path.relative_to(lineage_root)
    except ValueError as exc:
        raise DataReadinessError(
            "intraday parent contract snapshot is outside configs/lineage"
        ) from exc
    try:
        snapshot_file_sha256 = file_sha256(snapshot_path)
    except OSError as exc:
        raise DataReadinessError(
            f"intraday parent contract snapshot is unreadable: {snapshot_path}"
        ) from exc
    if snapshot_file_sha256 != expected_file_sha256:
        raise DataReadinessError(
            "intraday parent contract snapshot file hash does not match lineage"
        )
    snapshot_contract = load_strategy_contract(snapshot_path)
    if snapshot_contract.sha256() != expected_full_contract_sha256:
        raise DataReadinessError(
            "intraday parent contract snapshot full contract hash does not match lineage"
        )
    if (
        intraday_data_contract_sha256(snapshot_contract)
        != expected_intraday_sha256
    ):
        raise DataReadinessError(
            "intraday parent contract snapshot scoped hash does not match current scope"
        )


def _is_full_git_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _load_lineage(path: Path) -> dict[str, Any]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"intraday contract lineage is unreadable: {path}"
        ) from exc
    if raw.get("schema_version") != INTRADAY_CONTRACT_LINEAGE_SCHEMA:
        raise DataReadinessError("intraday contract lineage schema is invalid")
    return {str(key): value for key, value in raw.items()}
