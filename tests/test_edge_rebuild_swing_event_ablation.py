from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import swing_event_ablation as ablation
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PROFILE,
    swing_model_feature_columns,
)
from market_predictor.v3.errors import DataReadinessError

_ROOT = Path(__file__).parents[1]
_POLICY_PATH = _ROOT / "configs" / "swing_analyst_revision_ablation.toml"
_SECURITY_ID = "cik:0000000001:ticker:ACME"
_DECISION_TIME = pd.Timestamp("2021-07-10T20:00:00Z")


def test_policy_is_frozen_and_rejects_semantic_drift(tmp_path: Path) -> None:
    policy = ablation.load_analyst_revision_ablation_policy(_POLICY_PATH)

    assert policy.event_family == "analyst_revision"
    assert policy.source_family == "alpaca"
    assert policy.cohort_window == "3d"
    assert policy.near_window == "1d"
    assert policy.profiles == ablation.PROFILES
    assert policy.admitted_subtypes == (
        "bare_upgrade",
        "bare_downgrade",
        "coverage",
    )
    assert policy.directional_subtypes == ("bare_upgrade", "bare_downgrade")
    assert set(policy.diagnostic_only_subtypes) == {
        "price_target_up",
        "price_target_down",
        "analyst_rating_or_target_revision",
    }

    changed = tmp_path / "changed-policy.toml"
    changed.write_text(
        _POLICY_PATH.read_text(encoding="utf-8").replace(
            'unknown_coverage_policy = "abstain"',
            'unknown_coverage_policy = "known_zero"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataReadinessError, match="policy is not frozen"):
        ablation.load_analyst_revision_ablation_policy(changed)


def test_price_target_revision_remains_direction_unverified() -> None:
    policy = ablation.load_analyst_revision_ablation_policy(_POLICY_PATH)
    sources = _event_sources(
        matched_text="raises price target on acme",
        event_time=_DECISION_TIME - pd.Timedelta(hours=2),
        decision_time=_DECISION_TIME,
        include_near=True,
    )

    assert ablation._analyst_subtype("raises price target on acme") == (
        "direction_unverified"
    )
    row = ablation._build_event_features(sources, policy=policy).iloc[0]

    assert row["analyst_revision_present_1d"] == 1
    assert row["analyst_revision_latest_direction_unverified"] == 1
    assert row["analyst_revision_direction_available"] == 0
    assert row["analyst_revision_latest_is_upgrade"] == 0
    assert row["analyst_revision_latest_is_downgrade"] == 0
    assert row["analyst_revision_latest_is_coverage"] == 0

    assert ablation._analyst_subtype("upgrades acme price target") == (
        "direction_unverified"
    )
    assert ablation._analyst_subtype("initiated coverage on acme") == "coverage"


def test_weekend_revision_is_not_encoded_as_premarket() -> None:
    policy = ablation.load_analyst_revision_ablation_policy(_POLICY_PATH)
    decision_time = pd.Timestamp("2021-07-12T20:00:00Z")
    sources = _event_sources(
        matched_text="upgrades acme",
        event_time=pd.Timestamp("2021-07-10T11:00:00Z"),
        decision_time=decision_time,
    )

    row = ablation._build_event_features(sources, policy=policy).iloc[0]

    assert row["analyst_revision_latest_premarket"] == 0
    assert row["analyst_revision_latest_regular_session"] == 0
    assert row["analyst_revision_latest_after_close"] == 1


def test_exchange_holiday_revision_is_not_encoded_as_regular_session() -> None:
    assert ablation._publication_regime(
        pd.Timestamp("2021-07-05T16:00:00Z")
    ) == "after_close"


def test_contiguous_coverage_merges_across_historical_era_boundary() -> None:
    start = pd.Timestamp("2021-07-07T20:00:00Z")
    boundary = pd.Timestamp("2021-07-09T00:00:00Z")
    end = pd.Timestamp("2021-07-10T20:00:00Z")
    coverage = pd.DataFrame(
        [
            _coverage_row(start=start, end=boundary),
            _coverage_row(start=boundary, end=end),
        ]
    )

    intervals = ablation._merged_coverage_intervals(coverage)

    assert intervals[_SECURITY_ID] == ((start, end),)
    assert ablation._window_is_covered(
        intervals[_SECURITY_ID],
        start=start,
        end=end,
    )


@pytest.mark.parametrize(
    ("event_time", "accepted"),
    [
        (_DECISION_TIME - pd.Timedelta(days=3), True),
        (_DECISION_TIME, True),
        (_DECISION_TIME - pd.Timedelta(days=3, seconds=1), False),
        (_DECISION_TIME + pd.Timedelta(seconds=1), False),
    ],
)
def test_assignment_window_enforces_inclusive_causal_bounds(
    event_time: pd.Timestamp,
    accepted: bool,
) -> None:
    policy = ablation.load_analyst_revision_ablation_policy(_POLICY_PATH)
    sources = _event_sources(
        matched_text="upgrades acme",
        event_time=event_time,
        decision_time=_DECISION_TIME,
    )

    if accepted:
        result = ablation._build_event_features(sources, policy=policy)
        assert result["decision_id"].tolist() == ["decision-1"]
        assert result["analyst_revision_latest_age_fraction_3d"].between(0, 1).all()
    else:
        with pytest.raises(
            DataReadinessError,
            match="analyst assignment issuer or causal window fails",
        ):
            ablation._build_event_features(sources, policy=policy)


@pytest.mark.parametrize(
    ("missingness_known", "coverage_state"),
    [(False, "observed_complete"), (True, "failed")],
)
def test_unknown_or_failed_coverage_abstains_instead_of_becoming_zero(
    missingness_known: bool,
    coverage_state: str,
) -> None:
    policy = ablation.load_analyst_revision_ablation_policy(_POLICY_PATH)
    sources = _event_sources(
        matched_text="downgrades acme",
        event_time=_DECISION_TIME - pd.Timedelta(hours=3),
        decision_time=_DECISION_TIME,
        missingness_known=missingness_known,
        coverage_state=coverage_state,
    )

    assert ablation._merged_coverage_intervals(sources.coverage) == {}
    with pytest.raises(
        DataReadinessError,
        match="event feature identity is empty or duplicated",
    ):
        ablation._build_event_features(sources, policy=policy)


def test_profile_feature_projections_are_exact_and_column_disjoint() -> None:
    technical = ("return_5d_xs_z", "volume_z20_xs_rank")

    technical_only = ablation._profile_features(
        ablation.TECHNICAL_PROFILE,
        technical_features=technical,
    )
    event_only = ablation._profile_features(
        ablation.EVENT_PROFILE,
        technical_features=technical,
    )
    combined = ablation._profile_features(
        ablation.COMBINED_PROFILE,
        technical_features=technical,
    )

    assert technical_only == technical
    assert event_only == ablation.EVENT_FEATURE_COLUMNS
    assert set(technical_only).isdisjoint(event_only)
    assert combined == (*technical_only, *event_only)
    assert len(combined) == len(set(combined))
    with pytest.raises(DataReadinessError, match="unknown analyst ablation profile"):
        ablation._profile_features("catalyst_full", technical_features=technical)


def test_profile_identity_and_shared_content_hashes_detect_inequality_or_tampering() -> None:
    shared_columns = ("decision_id", "rank_label", "future_net_return_10d")
    reference = pd.DataFrame(
        {
            "decision_id": ["decision-1", "decision-2"],
            "rank_label": [1, -1],
            "future_net_return_10d": [0.04, -0.02],
        }
    )
    equal_copy = reference.copy()
    unequal_decisions = reference.iloc[:1].copy()
    tampered_label = reference.copy()
    tampered_label.loc[0, "rank_label"] = -1

    assert ablation._sequence_sha256(reference["decision_id"]) == (
        ablation._sequence_sha256(equal_copy["decision_id"])
    )
    assert ablation._frame_sha256(reference, shared_columns) == (
        ablation._frame_sha256(equal_copy, shared_columns)
    )
    assert ablation._sequence_sha256(reference["decision_id"]) != (
        ablation._sequence_sha256(unequal_decisions["decision_id"])
    )
    assert ablation._frame_sha256(reference, shared_columns) != (
        ablation._frame_sha256(tampered_label, shared_columns)
    )


def test_publish_replay_rejects_rebound_label_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_strategy_contract(
        _ROOT / "configs" / "edge_rebuild_strategy_contract.toml"
    )
    technical_features = swing_model_feature_columns(
        contract=contract,
        catalyst=False,
    )
    technical_dir = tmp_path / "technical"
    partition = technical_dir / "final" / "panel" / "part.parquet"
    partition.parent.mkdir(parents=True)
    row = _technical_row(technical_features, strategy_sha256=contract.sha256())
    pd.DataFrame([row]).to_parquet(partition, index=False)
    (technical_dir / "final" / "_manifest.json").write_text("{}", encoding="utf-8")
    (technical_dir / "final" / "_authority.json").write_text("{}", encoding="utf-8")
    panel = {
        "feature_profiles": [SWING_FEATURE_PROFILE],
        "strategy_contract_sha256": contract.sha256(),
        "files_by_profile": {
            SWING_FEATURE_PROFILE: [
                {"path": "panel/part.parquet", "partition_month": "2021-07"}
            ]
        },
    }
    sources = _bound_event_sources(tmp_path)
    monkeypatch.setattr(
        ablation,
        "load_complete_swing_feature_panel",
        lambda _directory: panel,
    )
    monkeypatch.setattr(
        ablation,
        "_load_event_sources",
        lambda *_args, **_kwargs: sources,
    )
    output = tmp_path / "ablation"

    published = ablation.publish_swing_analyst_revision_ablation(
        technical_panel_directory=technical_dir,
        event_authority_directories=[tmp_path / "event-1", tmp_path / "event-2"],
        precision_audit_directories=[tmp_path / "audit-1", tmp_path / "audit-2"],
        policy_path=_POLICY_PATH,
        strategy_contract=contract,
        output_directory=output,
    )

    assert published["rows_per_profile"] == 1
    assert ablation.load_swing_analyst_revision_ablation(
        output,
        strategy_contract=contract,
    )["episode_count"] == 1

    feature_tamper = tmp_path / "feature-tamper"
    shutil.copytree(output, feature_tamper)
    tampered_request_path = feature_tamper / "_request.json"
    tampered_request = json.loads(tampered_request_path.read_text(encoding="utf-8"))
    removed_feature = tampered_request["technical_feature_columns"].pop()
    tampered_request.pop("request_sha256")
    rebound_request_sha256 = ablation._json_sha256(tampered_request)
    tampered_request["request_sha256"] = rebound_request_sha256
    tampered_request_path.write_text(json.dumps(tampered_request), encoding="utf-8")
    tampered_manifest_path = feature_tamper / "_manifest.json"
    tampered_manifest = json.loads(
        tampered_manifest_path.read_text(encoding="utf-8")
    )
    tampered_manifest["request_sha256"] = rebound_request_sha256
    tampered_manifest["technical_feature_columns"].remove(removed_feature)
    for record in tampered_manifest["files"]:
        if removed_feature not in record["model_feature_columns"]:
            continue
        output_partition = feature_tamper / record["path"]
        frame = pd.read_parquet(output_partition).drop(columns=removed_feature)
        frame.to_parquet(output_partition, index=False)
        record["model_feature_columns"].remove(removed_feature)
        record["sha256"] = file_sha256(output_partition)
    tampered_manifest_path.write_text(
        json.dumps(tampered_manifest),
        encoding="utf-8",
    )
    tampered_authority_path = feature_tamper / "_authority.json"
    tampered_authority = json.loads(
        tampered_authority_path.read_text(encoding="utf-8")
    )
    tampered_authority["request_sha256"] = rebound_request_sha256
    tampered_authority["artifact_sha256"] = file_sha256(tampered_manifest_path)
    tampered_authority_path.write_text(
        json.dumps(tampered_authority),
        encoding="utf-8",
    )
    with pytest.raises(DataReadinessError, match="feature contract differs"):
        ablation.load_swing_analyst_revision_ablation(
            feature_tamper,
            strategy_contract=contract,
        )

    manifest_path = output / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        output_partition = output / record["path"]
        tampered = pd.read_parquet(output_partition)
        tampered.loc[0, "rank_label"] = -1
        tampered.to_parquet(output_partition, index=False)
        record["sha256"] = file_sha256(output_partition)
        record["shared_content_sha256"] = ablation._frame_sha256(
            tampered,
            tuple(manifest["shared_columns"]),
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = output / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="technical panel or event authority"):
        ablation.load_swing_analyst_revision_ablation(
            output,
            strategy_contract=contract,
        )


def _event_sources(
    *,
    matched_text: str,
    event_time: pd.Timestamp,
    decision_time: pd.Timestamp,
    include_near: bool = False,
    missingness_known: bool = True,
    coverage_state: str = "observed_complete",
) -> ablation._EventSources:
    event_id = "event-1"
    events = pd.DataFrame(
        {
            "family_event_id": [event_id],
            "security_id": [_SECURITY_ID],
            "classification_rule_id": ["analyst_rating_or_target_revision"],
            "matched_text": [matched_text],
            "feature_available_at_utc": [event_time],
        }
    )
    windows = ["3d", *(("1d",) if include_near else ())]
    assignments = pd.DataFrame(
        {
            "status": ["assigned"] * len(windows),
            "window_name": windows,
            "decision_time_utc": [decision_time] * len(windows),
            "feature_available_at_utc": [event_time] * len(windows),
            "event_id": [event_id] * len(windows),
            "security_id": [_SECURITY_ID] * len(windows),
            "decision_id": ["decision-1"] * len(windows),
        }
    )
    coverage = pd.DataFrame(
        [
            _coverage_row(
                start=decision_time - pd.Timedelta(days=3),
                end=decision_time,
                missingness_known=missingness_known,
                coverage_state=coverage_state,
            )
        ]
    )
    return ablation._EventSources(
        authorities=(),
        audits=(),
        events=events,
        assignments=assignments,
        coverage=coverage,
    )


def _bound_event_sources(tmp_path: Path) -> ablation._EventSources:
    raw = _event_sources(
        matched_text="upgrades acme",
        event_time=_DECISION_TIME - pd.Timedelta(hours=2),
        decision_time=_DECISION_TIME,
        include_near=True,
    )
    authorities = []
    audits = []
    for kind, target in (("event", authorities), ("audit", audits)):
        for index in (1, 2):
            directory = tmp_path / f"{kind}-{index}"
            directory.mkdir()
            (directory / "_authority.json").write_text("{}", encoding="utf-8")
            target.append(SimpleNamespace(directory=directory))
    return ablation._EventSources(
        authorities=tuple(authorities),
        audits=tuple(audits),
        events=raw.events,
        assignments=raw.assignments,
        coverage=raw.coverage,
    )


def _technical_row(
    technical_features: tuple[str, ...],
    *,
    strategy_sha256: str,
) -> dict[str, object]:
    timestamps = {
        column
        for column in ablation._IDENTITY_COLUMNS
        if column.endswith("_utc")
    }
    booleans = {
        column
        for column in ablation._IDENTITY_COLUMNS
        if column.endswith("_eligible")
    }
    row: dict[str, object] = {
        column: (
            _DECISION_TIME
            if column in timestamps
            else True
            if column in booleans
            else "value"
        )
        for column in ablation._IDENTITY_COLUMNS
    }
    row.update(
        {
            "decision_id": "decision-1",
            "decision_group_id": "2021-07-09",
            "ticker": "ACME",
            "security_id": _SECURITY_ID,
            "sector": "Information Technology",
            "primary_benchmark": "SPY",
            "session_date_et": "2021-07-09",
            "entry_price": 100.0,
            "exit_price": 104.0,
            "entry_session_date_et": "2021-07-12",
            "exit_session_date_et": "2021-07-23",
            "label_window_expected": True,
            "label_path_exact": True,
            "horizon_sessions": 10,
            "round_trip_cost_bps": 20.0,
            "minimum_daily_bars": 250,
            "dataset_label_config_sha256": "a" * 64,
            "execution_policy_sha256": "b" * 64,
            "strategy_contract_sha256": strategy_sha256,
            "rank_label": 1,
            "forward_return": 0.04,
        }
    )
    row.update({column: 0.1 for column in technical_features})
    return row


def _coverage_row(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    missingness_known: bool = True,
    coverage_state: str = "observed_complete",
) -> dict[str, object]:
    return {
        "security_id": _SECURITY_ID,
        "requested_start_utc": start,
        "requested_end_utc": end,
        "research_eligible": True,
        "missingness_known": missingness_known,
        "coverage_state": coverage_state,
    }
