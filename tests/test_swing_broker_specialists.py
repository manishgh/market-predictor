from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.edge_rebuild import swing_broker_specialists as specialists
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.v3.errors import DataReadinessError

_ROOT = Path(__file__).parents[1]
_POLICY = _ROOT / "configs" / "swing_broker_action_specialists.toml"


def test_policy_freezes_two_specialists_and_six_experiments_each() -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)

    assert policy.specialists == ("rating_change", "coverage_initiation")
    assert policy.profiles == (
        "technical_only",
        "broker_action_only",
        "technical_plus_broker_action",
    )
    assert policy.estimators == ("logistic", "hist_gradient_boosting")
    assert len(policy.profiles) * len(policy.estimators) == 6
    assert policy.minimum_validation_roc_auc == 0.60


def test_partial_policy_or_locked_test_date_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.toml"
    changed.write_text(
        _POLICY.read_text(encoding="utf-8").replace(
            "locked_test_start = 2025-07-01",
            "locked_test_start = 2025-06-30",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="policy is invalid"):
        specialists.load_broker_specialist_policy(changed)


def test_profile_records_never_open_locked_test_month() -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)
    manifest = {
        "files": [
            {
                "feature_profile": specialists.EVENT_PROFILE,
                "partition_month": "2025-06",
                "path": (
                    "panel/feature_profile=analyst_revision_event_only/"
                    "month=2025-06/part.parquet"
                ),
            },
            {
                "feature_profile": specialists.EVENT_PROFILE,
                "partition_month": "2025-07",
                "path": (
                    "panel/feature_profile=analyst_revision_event_only/"
                    "month=2025-07/part.parquet"
                ),
            },
        ]
    }

    records = specialists._profile_records(
        manifest,
        specialists.EVENT_PROFILE,
        policy,
    )

    assert [record["partition_month"] for record in records] == ["2025-06"]


def test_profile_record_rejects_path_month_mismatch_before_any_read() -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)
    manifest = {
        "files": [
            {
                "feature_profile": specialists.EVENT_PROFILE,
                "partition_month": "2025-06",
                "path": (
                    "panel/feature_profile=analyst_revision_event_only/"
                    "month=2025-07/part.parquet"
                ),
            }
        ]
    }

    with pytest.raises(DataReadinessError, match="partition identity is invalid"):
        specialists._profile_records(manifest, specialists.EVENT_PROFILE, policy)


def test_source_strategy_contract_drift_fails_closed() -> None:
    contract = load_strategy_contract(
        _ROOT / "configs" / "edge_rebuild_strategy_contract.toml"
    )

    with pytest.raises(DataReadinessError, match="source strategy contract differs"):
        specialists._assert_source_strategy_contract(
            {"strategy_contract_sha256": "0" * 64},
            contract,
        )


def test_output_control_flags_fail_closed() -> None:
    valid_authority = {
        "artifact": "_manifest.json",
        "locked_test_outcomes_read": False,
        "promotion_permitted": False,
    }
    valid_manifest = {
        "state": "complete",
        "locked_test_outcomes_read": False,
        "promotion_permitted": False,
    }

    specialists._assert_output_control_flags(valid_authority, valid_manifest)
    for field, invalid_value in (
        ("locked_test_outcomes_read", True),
        ("promotion_permitted", True),
    ):
        changed = dict(valid_authority)
        changed[field] = invalid_value
        with pytest.raises(DataReadinessError, match="control flags are invalid"):
            specialists._assert_output_control_flags(changed, valid_manifest)


def test_output_paths_cannot_escape_artifact(tmp_path: Path) -> None:
    expected = specialists._canonical_output_member(
        tmp_path,
        "capacity_audit.parquet",
        expected="capacity_audit.parquet",
    )

    assert expected == (tmp_path / "capacity_audit.parquet").resolve()
    with pytest.raises(DataReadinessError, match="output path is invalid"):
        specialists._canonical_output_member(
            tmp_path,
            "../capacity_audit.parquet",
            expected="capacity_audit.parquet",
        )


def test_capacity_audit_keeps_rating_and_coverage_populations_separate() -> None:
    policy = replace(
        specialists.load_broker_specialist_policy(_POLICY),
        minimum_development_announcements=1,
        minimum_validation_announcements=1,
        minimum_validation_securities=1,
        minimum_validation_sectors=1,
        minimum_unseen_validation_announcements=1,
    )
    frame = pd.DataFrame(
        {
            "decision_id": ["up-dev", "down-dev", "coverage-dev", "up-val", "down-val", "coverage-val"],
            "security_id": ["A", "B", "C", "D", "E", "F"],
            "sector": ["Tech", "Tech", "Health", "Tech", "Finance", "Health"],
            "session_date_et": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-07-02",
                    "2024-07-03",
                    "2024-07-04",
                ]
            ),
            "analyst_revision_episode_id": ["u1", "d1", "c1", "u2", "d2", "c2"],
            "subtype": [
                "rating_upgrade",
                "rating_downgrade",
                "coverage_initiation",
                "rating_upgrade",
                "rating_downgrade",
                "coverage_initiation",
            ],
        }
    )

    audit = specialists._capacity_audit(frame, policy)
    rating_development = audit.loc[
        audit["specialist"].eq("rating_change") & audit["split"].eq("development")
    ].iloc[0]
    coverage_development = audit.loc[
        audit["specialist"].eq("coverage_initiation")
        & audit["split"].eq("development")
    ].iloc[0]

    assert rating_development["announcements"] == 2
    assert rating_development["rating_up_announcements"] == 1
    assert rating_development["rating_down_announcements"] == 1
    assert coverage_development["announcements"] == 1
    assert coverage_development["coverage_announcements"] == 1


def test_coverage_is_an_action_even_when_direction_is_unavailable() -> None:
    flags = pd.DataFrame(
        {
            "analyst_revision_latest_is_upgrade": [1, 0, 0, 0],
            "analyst_revision_latest_is_downgrade": [0, 1, 0, 0],
            "analyst_revision_latest_is_coverage": [0, 0, 1, 0],
            "analyst_revision_latest_direction_unverified": [0, 0, 1, 1],
        }
    )

    assert specialists._classify_latest_subtypes(flags).tolist() == [
        "rating_upgrade",
        "rating_downgrade",
        "coverage_initiation",
        "price_target_or_generic",
    ]


def test_acceptance_requires_auc_and_brier_skill_in_both_scopes() -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)
    passing_scope = {
        "episode_weighted_roc_auc": 0.61,
        "episode_weighted_brier_skill_vs_train_prevalence": 0.01,
        "economic_gate": {"passed": True},
        "selected_unique_announcements": 200,
        "episode_weighted_expected_calibration_error": 0.05,
    }
    scopes = {
        "chronological_validation": passing_scope,
        "unseen_security_validation": {
            "episode_weighted_roc_auc": 0.59,
            "episode_weighted_brier_skill_vs_train_prevalence": 0.01,
            "economic_gate": {"passed": True},
            "selected_unique_announcements": 50,
            "episode_weighted_expected_calibration_error": 0.05,
        },
    }

    passed, reasons = specialists._acceptance_gate(scopes, policy)

    assert not passed
    assert reasons == ["unseen_security_validation ROC AUC is below 0.60"]


def test_acceptance_passes_only_when_predictive_and_economic_gates_pass() -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)
    scopes = {
        "chronological_validation": {
            "episode_weighted_roc_auc": 0.62,
            "episode_weighted_brier_skill_vs_train_prevalence": 0.02,
            "economic_gate": {"passed": True},
            "selected_unique_announcements": 200,
            "episode_weighted_expected_calibration_error": 0.05,
        },
        "unseen_security_validation": {
            "episode_weighted_roc_auc": 0.61,
            "episode_weighted_brier_skill_vs_train_prevalence": 0.01,
            "economic_gate": {"passed": True},
            "selected_unique_announcements": 50,
            "episode_weighted_expected_calibration_error": 0.05,
        },
    }

    assert specialists._acceptance_gate(scopes, policy) == (True, [])


def test_failed_inner_selection_never_loads_outer_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = specialists.load_broker_specialist_policy(_POLICY)
    loaded_end_dates: list[object] = []

    def fake_load(*_args: object, **kwargs: object) -> tuple[pd.DataFrame, tuple[str, ...]]:
        loaded_end_dates.append(kwargs["end_date"])
        return pd.DataFrame({"placeholder": [1]}), ("feature",)

    def fake_evaluate(
        _frame: pd.DataFrame,
        *,
        estimator_family: str,
        experiment_id: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "experiment_id": experiment_id,
            "profile": experiment_id.split(".")[1],
            "estimator_family": estimator_family,
            "feature_count": 1,
            "candidate_eligible": False,
            "outer_validation_opened": False,
            "locked_test_outcomes_read": False,
        }

    monkeypatch.setattr(specialists, "_load_training_profile", fake_load)
    monkeypatch.setattr(specialists, "_evaluate_experiment", fake_evaluate)
    monkeypatch.setattr(specialists, "_guard", lambda *_args, **_kwargs: None)

    result, model = specialists._run_specialist_experiments(
        source_directory=Path("unused"),
        source={},
        specialist="rating_change",
        decision_ids={"decision"},
        policy=policy,
        strategy_contract=object(),  # type: ignore[arg-type]
        swing_training_config=object(),  # type: ignore[arg-type]
    )

    assert model is None
    assert result["outer_validation_opened"] is False
    assert loaded_end_dates == [policy.development_end] * len(specialists.PROFILE_MAP)
