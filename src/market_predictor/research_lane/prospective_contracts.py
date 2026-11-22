"""Shared prospective candidate contracts, XNYS prediction-clock rules, and snapshot payload hashing.

Provides one immutable source of truth for:
1. Exchange-calendar (`XNYS`) session close, source cutoff, post-close settlement, and next-tradable-open
   timestamps (including early closes, weekends, and exchange holidays) (`R1`).
2. Immutable typed `ProspectiveCandidateSpec` binding model manifest SHA-256, ordered Stage-2 feature
   contract SHA-256, cohort, baseline, ranking population, required sources, horizon, selection,
   rebalance, cost, and missing-regime / OPRA observation policies (`R9`, `R6`).
3. Snapshot payload SHA-256 + byte-count verification (`ArtifactIntegrityError`) and prospective
   admission auditing (`R1`, `R9`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import exchange_calendars as xcals
import pandas as pd

from market_predictor.core.errors import ArtifactIntegrityError, DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research_lane.beta_neutral import PRECISE_STAGE2_FEATURE_NAMES
from market_predictor.research_lane.panel import file_sha256

RecordMode = Literal["prospective", "retrospective_replay"]
OpraRegimeState = Literal["backwardation", "contango", "unknown"]
OpraObservationPolicy = Literal["not_applicable", "synchronized_close_only", "allow_daily_vwap_proxy"]

XNYS_CALENDAR_NAME: Final = "XNYS"
DEFAULT_SETTLEMENT_DELAY_MINUTES: Final = 20

PROSPECTIVE_CANDIDATE_SCHEMA: Final = "market_predictor.research_lane.prospective_candidate.v1"
SNAPSHOT_MANIFEST_SCHEMA: Final = "market_predictor.research_lane.forward_snapshot.v2"

PERSISTED_SEC_MODEL_ID: Final = "mid_small_beta_neutral_sec_xgb"
PERSISTED_SEC_MODEL_MANIFEST_SHA256: Final = (
    "cd18356701f35b44c884d9f587a8e14ed7473878f12f75e2ec27548b4d8ebf04"
)
PERSISTED_SEC_FEATURE_CONTRACT_SHA256: Final = (
    "d244cea55bde7b79c22167da6bc83ad315f5165b33fbb3a39826b32d022b868e"
)

SNAPSHOT_KNOWN_PARQUET_PAYLOADS: Final[tuple[str, ...]] = (
    "scores.parquet",
    "holdings.parquet",
    "catalysts.parquet",
    "options_metrics.parquet",
)


@lru_cache(maxsize=1)
def get_xnys_calendar() -> Any:
    """Return cached `XNYS` exchange calendar."""
    return xcals.get_calendar(XNYS_CALENDAR_NAME)


@dataclass(frozen=True, slots=True)
class SessionClock:
    """Exact `XNYS` UTC timestamps governing decision cutoff, publication deadline, and next open."""

    as_of_date: date
    session_open_utc: datetime
    session_close_utc: datetime
    source_cutoff_utc: datetime
    settlement_ready_utc: datetime
    next_tradable_session: date
    next_tradable_open_utc: datetime
    is_early_close: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "calendar": XNYS_CALENDAR_NAME,
            "as_of_date": self.as_of_date.isoformat(),
            "session_open_utc": self.session_open_utc.isoformat(),
            "session_close_utc": self.session_close_utc.isoformat(),
            "source_cutoff_utc": self.source_cutoff_utc.isoformat(),
            "settlement_ready_utc": self.settlement_ready_utc.isoformat(),
            "next_tradable_session": self.next_tradable_session.isoformat(),
            "next_tradable_open_utc": self.next_tradable_open_utc.isoformat(),
            "is_early_close": self.is_early_close,
        }


def resolve_xnys_session_clock(
    as_of_date: date,
    *,
    settlement_delay_minutes: int = DEFAULT_SETTLEMENT_DELAY_MINUTES,
) -> SessionClock:
    """Compute exact `XNYS` close, source cutoff, settlement time, and next tradable session open."""
    cal = get_xnys_calendar()
    sess_ts = pd.Timestamp(as_of_date)
    if not bool(cal.is_session(sess_ts)):
        raise DataReadinessError(
            f"date {as_of_date.isoformat()} is not a valid {XNYS_CALENDAR_NAME} trading session"
        )
    s_open = pd.Timestamp(cal.session_open(sess_ts)).tz_convert("UTC").to_pydatetime()
    s_close = pd.Timestamp(cal.session_close(sess_ts)).tz_convert("UTC").to_pydatetime()
    next_sess_ts = cal.next_session(sess_ts)
    next_day = pd.Timestamp(next_sess_ts).date()
    next_open = pd.Timestamp(cal.session_open(next_sess_ts)).tz_convert("UTC").to_pydatetime()
    settlement = s_close + timedelta(minutes=int(settlement_delay_minutes))
    if not (s_close < settlement < next_open):
        raise DataReadinessError(
            f"settlement delay {settlement_delay_minutes}m violates post-close/pre-open interval for {as_of_date.isoformat()}"
        )
    early_close = (s_close - s_open) < timedelta(hours=6)
    return SessionClock(
        as_of_date=as_of_date,
        session_open_utc=s_open,
        session_close_utc=s_close,
        source_cutoff_utc=s_close,
        settlement_ready_utc=settlement,
        next_tradable_session=next_day,
        next_tradable_open_utc=next_open,
        is_early_close=early_close,
    )


def latest_settled_xnys_session(
    now_utc: datetime,
    *,
    settlement_delay_minutes: int = DEFAULT_SETTLEMENT_DELAY_MINUTES,
) -> SessionClock:
    """Return the latest `XNYS` session whose post-close settlement cutoff (`close + 20m`) has passed."""
    ts_utc = now_utc.astimezone(UTC) if now_utc.tzinfo is not None else now_utc.replace(tzinfo=UTC)
    cal = get_xnys_calendar()
    end_day = pd.Timestamp(ts_utc.date())
    start_day = end_day - pd.Timedelta(days=14)
    sessions = cal.sessions_in_range(start_day, end_day)
    settled: list[SessionClock] = []
    for sess_ts in sessions:
        clk = resolve_xnys_session_clock(
            pd.Timestamp(sess_ts).date(),
            settlement_delay_minutes=settlement_delay_minutes,
        )
        if clk.settlement_ready_utc <= ts_utc:
            settled.append(clk)
    if not settled:
        raise DataReadinessError(
            f"no settled {XNYS_CALENDAR_NAME} trading session found on or before {ts_utc.isoformat()}"
        )
    return settled[-1]


def parse_utc_timestamp(raw: object) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC `datetime`."""
    if raw is None or not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ProspectiveCandidateSpec:
    """Immutable specification binding model hashes, cohort, baseline, ranking population, and policy."""

    candidate_id: str
    cohort: Literal["mid_small_ex_sp500"]
    baseline: Literal["equal_rank_24_beta_neutral"]
    ranking_population: Literal["full_eligible_cohort"]
    min_warmup_sessions: int
    min_eligible_stocks: int
    required_sources: tuple[str, ...]
    expected_model_id: str
    expected_model_manifest_sha256: str
    expected_feature_contract_sha256: str
    feature_names: tuple[str, ...]
    score_column: str
    horizon_sessions: int
    core_spy_weight: float
    satellite_weight: float
    selection_rule: Literal["buffered_top_decile_equal_weight_buy10_hold33"]
    buy_top_fraction: float
    hold_top_fraction: float
    rebalance_rule: Literal["contiguous_open_to_open_every_horizon_sessions"]
    cost_bps: float
    regime_rule: Literal["none", "shift_satellite_to_spy_when_either_index_iv_term_30_90_gt_1"]
    missing_regime_policy: Literal["not_applicable", "abstain"]
    opra_observation_policy: OpraObservationPolicy
    schema: str = PROSPECTIVE_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if abs((self.core_spy_weight + self.satellite_weight) - 1.0) > 1e-12:
            raise ValueError(
                f"{self.candidate_id}: core_spy_weight + satellite_weight must equal 1.0"
            )
        if self.core_spy_weight < 0.0 or self.satellite_weight < 0.0:
            raise ValueError(f"{self.candidate_id}: negative portfolio sleeve weight prohibited")
        if self.horizon_sessions < 1:
            raise ValueError(f"{self.candidate_id}: horizon_sessions must be >= 1")
        if not (0.0 < self.buy_top_fraction <= self.hold_top_fraction <= 1.0):
            raise ValueError(f"{self.candidate_id}: invalid buy/hold fraction bounds")

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "cohort": self.cohort,
            "baseline": self.baseline,
            "ranking_population": self.ranking_population,
            "min_warmup_sessions": self.min_warmup_sessions,
            "min_eligible_stocks": self.min_eligible_stocks,
            "required_sources": list(self.required_sources),
            "expected_model_id": self.expected_model_id,
            "expected_model_manifest_sha256": self.expected_model_manifest_sha256,
            "expected_feature_contract_sha256": self.expected_feature_contract_sha256,
            "feature_names": list(self.feature_names),
            "score_column": self.score_column,
            "horizon_sessions": self.horizon_sessions,
            "core_spy_weight": self.core_spy_weight,
            "satellite_weight": self.satellite_weight,
            "selection_rule": self.selection_rule,
            "buy_top_fraction": round(self.buy_top_fraction, 12),
            "hold_top_fraction": round(self.hold_top_fraction, 12),
            "rebalance_rule": self.rebalance_rule,
            "cost_bps": self.cost_bps,
            "regime_rule": self.regime_rule,
            "missing_regime_policy": self.missing_regime_policy,
            "opra_observation_policy": self.opra_observation_policy,
        }

    @property
    def candidate_spec_sha256(self) -> str:
        return json_sha256(self.payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_without_hash()
        payload["candidate_spec_sha256"] = self.candidate_spec_sha256
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ProspectiveCandidateSpec:
        required_keys = {
            "candidate_id",
            "cohort",
            "baseline",
            "ranking_population",
            "min_warmup_sessions",
            "min_eligible_stocks",
            "required_sources",
            "expected_model_id",
            "expected_model_manifest_sha256",
            "expected_feature_contract_sha256",
            "feature_names",
            "score_column",
            "horizon_sessions",
            "core_spy_weight",
            "satellite_weight",
            "selection_rule",
            "buy_top_fraction",
            "hold_top_fraction",
            "rebalance_rule",
            "cost_bps",
            "regime_rule",
            "missing_regime_policy",
            "opra_observation_policy",
        }
        missing = sorted(required_keys - set(raw.keys()))
        if missing:
            raise ValueError(f"incomplete ProspectiveCandidateSpec missing fields: {missing}")
        spec = cls(
            schema=str(raw.get("schema") or PROSPECTIVE_CANDIDATE_SCHEMA),
            candidate_id=str(raw["candidate_id"]),
            cohort=raw["cohort"],
            baseline=raw["baseline"],
            ranking_population=raw["ranking_population"],
            min_warmup_sessions=int(raw["min_warmup_sessions"]),
            min_eligible_stocks=int(raw["min_eligible_stocks"]),
            required_sources=tuple(str(x) for x in raw["required_sources"]),
            expected_model_id=str(raw["expected_model_id"]),
            expected_model_manifest_sha256=str(raw["expected_model_manifest_sha256"]),
            expected_feature_contract_sha256=str(raw["expected_feature_contract_sha256"]),
            feature_names=tuple(str(x) for x in raw["feature_names"]),
            score_column=str(raw["score_column"]),
            horizon_sessions=int(raw["horizon_sessions"]),
            core_spy_weight=float(raw["core_spy_weight"]),
            satellite_weight=float(raw["satellite_weight"]),
            selection_rule=raw["selection_rule"],
            buy_top_fraction=float(raw["buy_top_fraction"]),
            hold_top_fraction=float(raw["hold_top_fraction"]),
            rebalance_rule=raw["rebalance_rule"],
            cost_bps=float(raw["cost_bps"]),
            regime_rule=raw["regime_rule"],
            missing_regime_policy=raw["missing_regime_policy"],
            opra_observation_policy=raw["opra_observation_policy"],
        )
        claimed_sha = raw.get("candidate_spec_sha256")
        if claimed_sha is not None and str(claimed_sha) != spec.candidate_spec_sha256:
            raise ArtifactIntegrityError(
                f"candidate_spec_sha256 mismatch on {spec.candidate_id}: "
                f"expected {spec.candidate_spec_sha256}, got {claimed_sha}"
            )
        return spec


_SEC_BASE_SOURCES: Final[tuple[str, ...]] = (
    "alpaca_sip_daily_adjusted",
    "alpaca_sip_daily_unadjusted",
    "sp500_point_in_time_membership",
    "wider_us_universe_roster_spans",
    "sec_company_facts",
    "sec_edgar_submissions",
)

FROZEN_PROSPECTIVE_CANDIDATE_SPECS: Final[tuple[ProspectiveCandidateSpec, ...]] = (
    ProspectiveCandidateSpec(
        candidate_id="mid_small_bn_sec_xgb_10d",
        cohort="mid_small_ex_sp500",
        baseline="equal_rank_24_beta_neutral",
        ranking_population="full_eligible_cohort",
        min_warmup_sessions=252,
        min_eligible_stocks=30,
        required_sources=_SEC_BASE_SOURCES,
        expected_model_id=PERSISTED_SEC_MODEL_ID,
        expected_model_manifest_sha256=PERSISTED_SEC_MODEL_MANIFEST_SHA256,
        expected_feature_contract_sha256=PERSISTED_SEC_FEATURE_CONTRACT_SHA256,
        feature_names=PRECISE_STAGE2_FEATURE_NAMES,
        score_column="score_mid_small_bn_sec_xgb_10d",
        horizon_sessions=10,
        core_spy_weight=0.0,
        satellite_weight=1.0,
        selection_rule="buffered_top_decile_equal_weight_buy10_hold33",
        buy_top_fraction=0.10,
        hold_top_fraction=1.0 / 3.0,
        rebalance_rule="contiguous_open_to_open_every_horizon_sessions",
        cost_bps=20.0,
        regime_rule="none",
        missing_regime_policy="not_applicable",
        opra_observation_policy="not_applicable",
    ),
    ProspectiveCandidateSpec(
        candidate_id="mid_small_bn_sec_xgb_21d",
        cohort="mid_small_ex_sp500",
        baseline="equal_rank_24_beta_neutral",
        ranking_population="full_eligible_cohort",
        min_warmup_sessions=252,
        min_eligible_stocks=30,
        required_sources=_SEC_BASE_SOURCES,
        expected_model_id=PERSISTED_SEC_MODEL_ID,
        expected_model_manifest_sha256=PERSISTED_SEC_MODEL_MANIFEST_SHA256,
        expected_feature_contract_sha256=PERSISTED_SEC_FEATURE_CONTRACT_SHA256,
        feature_names=PRECISE_STAGE2_FEATURE_NAMES,
        score_column="score_mid_small_bn_sec_xgb_21d",
        horizon_sessions=21,
        core_spy_weight=0.0,
        satellite_weight=1.0,
        selection_rule="buffered_top_decile_equal_weight_buy10_hold33",
        buy_top_fraction=0.10,
        hold_top_fraction=1.0 / 3.0,
        rebalance_rule="contiguous_open_to_open_every_horizon_sessions",
        cost_bps=20.0,
        regime_rule="none",
        missing_regime_policy="not_applicable",
        opra_observation_policy="not_applicable",
    ),
    ProspectiveCandidateSpec(
        candidate_id="mid_small_bn_sec_xgb_opra_10d",
        cohort="mid_small_ex_sp500",
        baseline="equal_rank_24_beta_neutral",
        ranking_population="full_eligible_cohort",
        min_warmup_sessions=252,
        min_eligible_stocks=30,
        required_sources=(*_SEC_BASE_SOURCES, "alpaca_opra_synchronized_close_options"),
        expected_model_id=PERSISTED_SEC_MODEL_ID,
        expected_model_manifest_sha256=PERSISTED_SEC_MODEL_MANIFEST_SHA256,
        expected_feature_contract_sha256=PERSISTED_SEC_FEATURE_CONTRACT_SHA256,
        feature_names=PRECISE_STAGE2_FEATURE_NAMES,
        score_column="score_mid_small_bn_sec_xgb_opra_10d",
        horizon_sessions=10,
        core_spy_weight=0.90,
        satellite_weight=0.10,
        selection_rule="buffered_top_decile_equal_weight_buy10_hold33",
        buy_top_fraction=0.10,
        hold_top_fraction=1.0 / 3.0,
        rebalance_rule="contiguous_open_to_open_every_horizon_sessions",
        cost_bps=20.0,
        regime_rule="shift_satellite_to_spy_when_either_index_iv_term_30_90_gt_1",
        missing_regime_policy="abstain",
        opra_observation_policy="synchronized_close_only",
    ),
)

FROZEN_PROSPECTIVE_CANDIDATE_MAP: Final[dict[str, ProspectiveCandidateSpec]] = {
    spec.candidate_id: spec for spec in FROZEN_PROSPECTIVE_CANDIDATE_SPECS
}


def build_payload_artifacts_manifest(staging_dir: Path) -> dict[str, dict[str, Any]]:
    """Compute `{filename: {'bytes': ..., 'sha256': ...}}` for every parquet payload in `staging_dir`."""
    artifacts: dict[str, dict[str, Any]] = {}
    for fname in SNAPSHOT_KNOWN_PARQUET_PAYLOADS:
        fpath = staging_dir / fname
        if fpath.is_file():
            artifacts[fname] = {
                "bytes": int(fpath.stat().st_size),
                "sha256": file_sha256(fpath),
            }
    if "scores.parquet" not in artifacts:
        raise ArtifactIntegrityError(f"snapshot staging directory missing required scores.parquet: {staging_dir}")
    return artifacts


def verify_snapshot_payload_hashes(snapshot_dir: Path, manifest: Mapping[str, Any]) -> Literal["verified", "legacy_unhashed"]:
    """Verify snapshot payload SHA-256s and byte sizes; raise `ArtifactIntegrityError` on any tamper."""
    raw_artifacts = manifest.get("payload_artifacts")
    if raw_artifacts is None:
        return "legacy_unhashed"
    if not isinstance(raw_artifacts, Mapping) or "scores.parquet" not in raw_artifacts:
        raise ArtifactIntegrityError(
            f"snapshot manifest in {snapshot_dir} has malformed payload_artifacts"
        )
    for fname, meta in raw_artifacts.items():
        if not isinstance(meta, Mapping):
            raise ArtifactIntegrityError(f"invalid payload metadata for {fname} in {snapshot_dir}")
        fpath = snapshot_dir / str(fname)
        if not fpath.is_file():
            raise ArtifactIntegrityError(f"missing declared snapshot payload file: {fpath}")
        actual_bytes = int(fpath.stat().st_size)
        expected_bytes = int(meta.get("bytes", -1))
        if actual_bytes != expected_bytes:
            raise ArtifactIntegrityError(
                f"byte-size mismatch on {fpath}: expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha = file_sha256(fpath)
        expected_sha = str(meta.get("sha256", ""))
        if actual_sha != expected_sha:
            raise ArtifactIntegrityError(
                f"SHA-256 tamper mismatch on {fpath}: expected {expected_sha}, got {actual_sha}"
            )
    for fname in SNAPSHOT_KNOWN_PARQUET_PAYLOADS:
        if (snapshot_dir / fname).is_file() and fname not in raw_artifacts:
            raise ArtifactIntegrityError(
                f"untracked payload file {fname} present in {snapshot_dir} but absent from _manifest.json"
            )
    return "verified"


def audit_snapshot_admission(
    snapshot_dir: Path,
    scores_df: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    expected_candidates: tuple[ProspectiveCandidateSpec, ...] = FROZEN_PROSPECTIVE_CANDIDATE_SPECS,
    allow_retrospective_replay: bool = False,
) -> dict[str, Any]:
    """Audit prediction clock, payload hashes, cohort, model identity, and candidate admission."""
    snap_day = date.fromisoformat(snapshot_dir.name)
    scored_session = str(manifest.get("scored_session") or snap_day.isoformat())
    if scored_session != snap_day.isoformat():
        raise ArtifactIntegrityError(
            f"snapshot directory {snapshot_dir.name} disagrees with manifest scored_session {scored_session}"
        )

    clock = resolve_xnys_session_clock(snap_day)
    payload_status = verify_snapshot_payload_hashes(snapshot_dir, manifest)

    pub_utc = parse_utc_timestamp(manifest.get("published_at_utc") or manifest.get("recorded_at_utc"))
    raw_mode = manifest.get("record_mode")
    if raw_mode in ("prospective", "retrospective_replay"):
        record_mode = str(raw_mode)
    else:
        record_mode = "legacy_untyped"

    if pub_utc is None:
        published_before_entry = False
        clock_status = "missing_publication_timestamp"
    elif pub_utc < clock.next_tradable_open_utc:
        published_before_entry = True
        clock_status = "published_before_entry"
    else:
        published_before_entry = False
        clock_status = "recorded_after_entry"

    cat_sum = manifest.get("catalyst_summary")
    sec_model_meta = cat_sum.get("sec_xgb_model") if isinstance(cat_sum, Mapping) else None
    actual_model_sha = (
        str(sec_model_meta.get("manifest_sha256") or "")
        if isinstance(sec_model_meta, Mapping)
        else ""
    )
    actual_feat_sha = (
        str(sec_model_meta.get("feature_contract_sha256") or "")
        if isinstance(sec_model_meta, Mapping)
        else ""
    )
    sec_model_scored = bool(isinstance(sec_model_meta, Mapping) and sec_model_meta.get("model_scored") is True)

    # Parse persisted candidate definitions inside the snapshot manifest
    snap_specs_by_id: dict[str, ProspectiveCandidateSpec] = {}
    incomplete_spec_reasons: dict[str, str] = {}
    raw_cands = manifest.get("frozen_prospective_candidates")
    if isinstance(raw_cands, list):
        for item in raw_cands:
            if isinstance(item, Mapping) and "candidate_id" in item:
                cid = str(item["candidate_id"])
                try:
                    snap_specs_by_id[cid] = ProspectiveCandidateSpec.from_mapping(item)
                except ValueError as exc:
                    incomplete_spec_reasons[cid] = str(exc)

    sectors_series = scores_df["sector"].fillna("Unknown").astype(str) if "sector" in scores_df.columns else pd.Series(dtype=str)
    unknown_sector_rows = int((sectors_series == "Unknown").sum()) if len(sectors_series) else len(scores_df)
    all_cik_ids = (
        bool(scores_df["security_id"].astype(str).str.startswith("cik:").all())
        if len(scores_df) and "security_id" in scores_df.columns
        else False
    )

    candidate_admissions: dict[str, dict[str, Any]] = {}
    for expected in expected_candidates:
        reasons: list[str] = []
        if record_mode == "retrospective_replay":
            if not allow_retrospective_replay:
                reasons.append("retrospective_replay_excluded_from_prospective_pool")
        elif record_mode != "prospective":
            reasons.append(f"untyped_record_mode:{record_mode}")

        if record_mode != "retrospective_replay" and not published_before_entry:
            reasons.append(clock_status)

        if payload_status != "verified":
            reasons.append(f"payload_hashes_not_verified:{payload_status}")

        if expected.candidate_id in incomplete_spec_reasons:
            reasons.append(f"incomplete_frozen_candidate_spec:{incomplete_spec_reasons[expected.candidate_id]}")
        elif expected.candidate_id not in snap_specs_by_id:
            reasons.append("candidate_not_declared_in_snapshot")
        else:
            persisted_spec = snap_specs_by_id[expected.candidate_id]
            if persisted_spec.candidate_spec_sha256 != expected.candidate_spec_sha256:
                reasons.append(
                    f"candidate_spec_mismatch:{persisted_spec.candidate_spec_sha256}!={expected.candidate_spec_sha256}"
                )

        if not all_cik_ids or unknown_sector_rows > 0:
            reasons.append("cohort_identity_or_sector_not_verified")

        if not sec_model_scored:
            abstain_detail = (
                str(sec_model_meta.get("reason"))
                if isinstance(sec_model_meta, Mapping) and sec_model_meta.get("reason")
                else "sec_xgb_model_not_scored"
            )
            reasons.append(f"missing_sec_model:{abstain_detail}")
        else:
            if actual_model_sha != expected.expected_model_manifest_sha256:
                reasons.append(
                    f"model_manifest_sha256_mismatch:{actual_model_sha}!={expected.expected_model_manifest_sha256}"
                )
            if actual_feat_sha != expected.expected_feature_contract_sha256:
                reasons.append(
                    f"feature_contract_sha256_mismatch:{actual_feat_sha}!={expected.expected_feature_contract_sha256}"
                )

        if expected.score_column not in scores_df.columns:
            reasons.append(f"missing_score_column:{expected.score_column}")
        elif int(pd.to_numeric(scores_df[expected.score_column], errors="coerce").notna().sum()) < expected.min_eligible_stocks:
            reasons.append(f"insufficient_scored_stocks_in_{expected.score_column}")

        if not (snapshot_dir / "holdings.parquet").is_file():
            reasons.append("missing_persisted_holdings_parquet")

        candidate_admissions[expected.candidate_id] = {
            "admitted": len(reasons) == 0,
            "rejection_reasons": reasons,
        }

    sec_primary_admitted = any(
        candidate_admissions[cid]["admitted"]
        for cid in ("mid_small_bn_sec_xgb_10d", "mid_small_bn_sec_xgb_21d", "mid_small_bn_sec_xgb_opra_10d")
        if cid in candidate_admissions
    )
    return {
        "snapshot_date": snap_day.isoformat(),
        "record_mode": record_mode,
        "source_cutoff_utc": str(manifest.get("source_cutoff_utc") or clock.source_cutoff_utc.isoformat()),
        "published_at_utc": pub_utc.isoformat() if pub_utc is not None else None,
        "next_tradable_session": clock.next_tradable_session.isoformat(),
        "next_tradable_open_utc": clock.next_tradable_open_utc.isoformat(),
        "is_early_close": clock.is_early_close,
        "published_before_entry": published_before_entry,
        "clock_status": clock_status,
        "payload_integrity_status": payload_status,
        "all_cik_security_ids": all_cik_ids,
        "unknown_sector_rows": unknown_sector_rows,
        "sec_model_scored": sec_model_scored,
        "sec_model_manifest_sha256": actual_model_sha or None,
        "qualifies_for_persisted_sec_model": sec_primary_admitted,
        "candidate_admissions": candidate_admissions,
    }


__all__ = [
    "DEFAULT_SETTLEMENT_DELAY_MINUTES",
    "FROZEN_PROSPECTIVE_CANDIDATE_MAP",
    "FROZEN_PROSPECTIVE_CANDIDATE_SPECS",
    "OpraObservationPolicy",
    "OpraRegimeState",
    "PERSISTED_SEC_FEATURE_CONTRACT_SHA256",
    "PERSISTED_SEC_MODEL_ID",
    "PERSISTED_SEC_MODEL_MANIFEST_SHA256",
    "PROSPECTIVE_CANDIDATE_SCHEMA",
    "ProspectiveCandidateSpec",
    "RecordMode",
    "SNAPSHOT_KNOWN_PARQUET_PAYLOADS",
    "SNAPSHOT_MANIFEST_SCHEMA",
    "SessionClock",
    "XNYS_CALENDAR_NAME",
    "audit_snapshot_admission",
    "build_payload_artifacts_manifest",
    "get_xnys_calendar",
    "latest_settled_xnys_session",
    "parse_utc_timestamp",
    "resolve_xnys_session_clock",
    "verify_snapshot_payload_hashes",
]
