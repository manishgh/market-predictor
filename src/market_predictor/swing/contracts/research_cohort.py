"""Whole-security research restrictions, separate from source and return admission."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SecurityId = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]


class ResearchSecurityExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    security_id: SecurityId
    tickers: tuple[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,15}$")], ...] = Field(min_length=1)
    reason: Literal["unresolved_holding_identity", "adjusted_price_mismatch", "unresolved_corporate_action", "unavailable_trading"]


class SwingResearchCohort(BaseModel):
    """A frozen retrospective population; never an accounting or promotion pass.

    The original population excludes warm-up-only securities. Cumulative exclusions
    include inherited coverage failures even when provider bars are already absent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal["market_predictor.swing_research_cohort"]
    scope: Literal["retrospective_development_restriction"]
    price_basis_status: Literal["not_certified_by_cohort"]
    combined_daily_inputs_sha256: Sha256
    original_security_ids: tuple[SecurityId, ...] = Field(min_length=1)
    inherited_excluded_security_ids: tuple[SecurityId, ...]
    warmup_only_security_ids: tuple[SecurityId, ...]
    exclusions: tuple[ResearchSecurityExclusion, ...] = Field(min_length=1)
    maximum_exclusion_bps: int = Field(ge=0, le=10000)
    cap_approval_reference: str = Field(min_length=1)
    source_files: dict[str, Sha256] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        for values in (self.original_security_ids, self.inherited_excluded_security_ids, self.warmup_only_security_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("cohort security lists must be unique and sorted")
        original = set(self.original_security_ids)
        inherited = set(self.inherited_excluded_security_ids)
        added = [entry.security_id for entry in self.exclusions]
        if added != sorted(set(added)):
            raise ValueError("cohort exclusions must be unique and sorted by security_id")
        if not inherited.issubset(original) or not set(added).issubset(original):
            raise ValueError("cohort exclusion references an unknown modeled security")
        if original.intersection(self.warmup_only_security_ids):
            raise ValueError("warm-up-only securities cannot enter the exclusion denominator")
        if not self.retained_security_ids:
            raise ValueError("cohort cannot exclude the complete population")
        for entry in self.exclusions:
            if entry.tickers != tuple(sorted(set(entry.tickers))):
                raise ValueError("cohort ticker displays must be unique and sorted")
        return self

    @property
    def excluded_security_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.inherited_excluded_security_ids).union(e.security_id for e in self.exclusions)))

    @property
    def retained_security_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.original_security_ids).difference(self.excluded_security_ids)))

    @property
    def within_cap(self) -> bool:
        return len(self.excluded_security_ids) * 10000 <= self.maximum_exclusion_bps * len(self.original_security_ids)

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))

    def summary(self) -> dict[str, Any]:
        return {
            "status": "accepted_research_restriction" if self.within_cap else "blocked_exclusion_cap",
            "original_securities": len(self.original_security_ids),
            "inherited_excluded_securities": len(self.inherited_excluded_security_ids),
            "additional_excluded_securities": len(self.excluded_security_ids) - len(self.inherited_excluded_security_ids),
            "total_excluded_securities": len(self.excluded_security_ids),
            "retained_securities": len(self.retained_security_ids),
            "excluded_fraction": len(self.excluded_security_ids) / len(self.original_security_ids),
            "maximum_exclusion_bps": self.maximum_exclusion_bps,
            "accounting_eligible": False,
            "promotion_eligible": False,
        }

    def assert_source_matches(
        self, combined_inputs: Mapping[str, Any], retained_ids: tuple[str, ...], warmup_only_ids: tuple[str, ...],
    ) -> None:
        """Reject changed parent populations, inherited exclusions or provider lineage."""
        expected_parent = tuple(sorted(set(self.original_security_ids).difference(self.inherited_excluded_security_ids)))
        if (
            json_sha256(dict(combined_inputs)) != self.combined_daily_inputs_sha256
            or combined_inputs.get("modeled_security_count") != len(self.original_security_ids)
            or combined_inputs.get("modeled_security_ids_sha256") != json_sha256(list(self.original_security_ids))
            or combined_inputs.get("excluded_security_ids") != list(self.inherited_excluded_security_ids)
            or combined_inputs.get("excluded_security_ids_sha256") != json_sha256(list(self.inherited_excluded_security_ids))
            or combined_inputs.get("excluded_security_count") != len(self.inherited_excluded_security_ids)
            or combined_inputs.get("retained_security_count") != len(expected_parent)
            or retained_ids != expected_parent
            or warmup_only_ids != self.warmup_only_security_ids
        ):
            raise DataReadinessError("research cohort parent population or source binding differs")

    def restrict_memberships(self, memberships: pd.DataFrame, combined_inputs: Mapping[str, Any]) -> pd.DataFrame:
        """Apply before bar batches, peer transforms, labels, and selection, never after."""
        if "security_id" not in memberships or memberships["security_id"].isna().any():
            raise DataReadinessError("research cohort requires non-null security identities")
        ids = tuple(sorted(memberships["security_id"].unique()))
        self.assert_source_matches(combined_inputs, ids, self.warmup_only_security_ids)
        if not self.within_cap:
            raise DataReadinessError(
                f"research exclusions exceed approved cap: {len(self.excluded_security_ids)}/"
                f"{len(self.original_security_ids)} versus {self.maximum_exclusion_bps / 100:.2f}%"
            )
        return memberships.loc[memberships["security_id"].isin(self.retained_security_ids)].copy()


def load_swing_research_cohort(path: Path, *, source_root: Path | None = None) -> SwingResearchCohort:
    """Read a hash-bound audit and optionally verify its immutable source files."""
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise DataReadinessError("research cohort metadata exceeds bounded size")
        payload = parse_strict_json_object(path.read_bytes(), label=str(path))
        if set(payload) != {"cohort", "cohort_sha256", "summary", "coverage", "audit_sha256"}:
            raise DataReadinessError("research cohort envelope fields differ")
        if payload["audit_sha256"] != json_sha256({key: value for key, value in payload.items() if key != "audit_sha256"}):
            raise DataReadinessError("research cohort audit hash differs")
        cohort = SwingResearchCohort.model_validate_json(json.dumps(payload["cohort"], allow_nan=False))
        if payload["cohort_sha256"] != cohort.sha256() or payload["summary"] != cohort.summary():
            raise DataReadinessError("research cohort identity or summary differs")
        if source_root is not None:
            for relative, expected in cohort.source_files.items():
                if file_sha256(resolve_inside_authority(source_root, relative)) != expected:
                    raise DataReadinessError(f"research cohort source hash differs: {relative}")
        return cohort
    except (OSError, ValueError, TypeError) as exc:
        raise DataReadinessError(f"invalid research cohort: {path}: {exc}") from exc
