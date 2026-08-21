from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.edge_rebuild import intraday_event_training as training
from market_predictor.core.errors import DataReadinessError


def test_proxy_event_cohort_is_deduplicated_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "_authority.json").write_text("{}", encoding="utf-8")
    (root / "_manifest.json").write_text("{}", encoding="utf-8")
    event_ids = [f"event-{index}" for index in range(1_000)]
    attachments = pd.DataFrame(
        {
            "family_event_id": event_ids,
            "decision_id": ["decision-1"] * 500 + ["decision-2"] * 500,
            "research_eligible": True,
            "identity_alignment": "exact_ticker_cik_compatible",
            "feature_available_at_utc": pd.Timestamp("2025-01-02T14:00:00Z"),
            "decision_time_utc": pd.Timestamp("2025-01-02T15:00:00Z"),
        }
    )
    authority = SimpleNamespace(
        manifest={
            "status": "blocked",
            "training_eligible": False,
            "serving_eligible": False,
            "future_holdout_opened": False,
            "request_sha256": "1" * 64,
            "blockers": sorted(training.EXPECTED_PROXY_BLOCKERS),
        },
        attachments=attachments,
    )
    monkeypatch.setattr(
        training,
        "load_intraday_event_preflight",
        lambda _directory, **_kwargs: authority,
    )

    cohort = training.load_intraday_research_event_cohort(root)
    frame = pd.DataFrame(
        {
            "decision_id": ["decision-1", "decision-2", "decision-3"],
            "value": [1, 2, 3],
        }
    )
    filtered = training.filter_to_research_event_cohort(frame, cohort)

    assert cohort.decision_ids == frozenset({"decision-1", "decision-2"})
    assert cohort.identity["event_episodes"] == 1_000
    assert cohort.identity["production_eligible"] is False
    assert filtered["decision_id"].tolist() == ["decision-1", "decision-2"]


def test_proxy_event_cohort_rejects_future_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "_authority.json").write_text("{}", encoding="utf-8")
    (root / "_manifest.json").write_text("{}", encoding="utf-8")
    attachments = pd.DataFrame(
        {
            "family_event_id": [f"event-{index}" for index in range(1_000)],
            "decision_id": [f"decision-{index}" for index in range(1_000)],
            "research_eligible": True,
            "identity_alignment": "exact_ticker_cik_compatible",
            "feature_available_at_utc": pd.Timestamp("2025-01-02T16:00:00Z"),
            "decision_time_utc": pd.Timestamp("2025-01-02T15:00:00Z"),
        }
    )
    monkeypatch.setattr(
        training,
        "load_intraday_event_preflight",
        lambda _directory, **_kwargs: SimpleNamespace(
            manifest={
                "status": "blocked",
                "training_eligible": False,
                "serving_eligible": False,
                "future_holdout_opened": False,
                "blockers": sorted(training.EXPECTED_PROXY_BLOCKERS),
            },
            attachments=attachments,
        ),
    )

    with pytest.raises(DataReadinessError, match="future evidence"):
        training.load_intraday_research_event_cohort(root)


def test_directional_upgrade_cohort_enforces_and_records_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "_authority.json").write_text("{}", encoding="utf-8")
    (root / "_manifest.json").write_text("{}", encoding="utf-8")
    event_ids = [f"event-{index}" for index in range(500)]
    decision_ids = [f"decision-{index}" for index in range(500)]
    folds = [-1] * 100 + [fold for fold in range(4) for _ in range(100)]
    security_ids = [f"cik:{index % 250:010d}" for index in range(500)]
    attachments = pd.DataFrame(
        {
            "family_event_id": event_ids,
            "decision_id": decision_ids,
            "security_id": security_ids,
            "research_eligible": True,
            "identity_alignment": "exact_ticker_cik_compatible",
            "feature_available_at_utc": pd.Timestamp("2025-01-02T14:00:00Z"),
            "decision_time_utc": pd.Timestamp("2025-01-02T15:00:00Z"),
        }
    )
    decisions = pd.DataFrame(
        {
            "decision_id": decision_ids,
            "security_id": security_ids,
            "session_date_et": pd.date_range("2023-01-02", periods=500, freq="D").date,
            "development_fold": folds,
            "validation_scope": ["unseen_security"] * 100 + ["seen_security"] * 400,
        }
    )
    authority = SimpleNamespace(
        manifest={
            "status": "blocked",
            "training_eligible": False,
            "serving_eligible": False,
            "future_holdout_opened": False,
            "request_sha256": "1" * 64,
            "blockers": sorted(training.EXPECTED_PROXY_BLOCKERS),
            "event_authorities": [{"directory": "bound-parent"}],
        },
        attachments=attachments,
        decisions=decisions,
    )
    monkeypatch.setattr(
        training,
        "load_intraday_event_preflight",
        lambda _directory, **_kwargs: authority,
    )
    monkeypatch.setattr(
        training,
        "_load_event_subtypes",
        lambda _manifest, **_kwargs: pd.DataFrame(
            {"family_event_id": event_ids, "event_subtype": "bare_upgrade"}
        ),
    )

    cohort = training.load_intraday_research_event_cohort(
        root,
        event_subtype="bare_upgrade",
    )

    assert len(cohort.decision_ids) == 500
    assert cohort.identity["event_subtype"] == "bare_upgrade"
    assert cohort.identity["directional_capacity"] == {
        "event_episodes": 500,
        "decision_rows": 500,
        "securities": 250,
        "sessions": 500,
        "events_by_validation_fold": {"0": 100, "1": 100, "2": 100, "3": 100},
        "unseen_security_events": 100,
    }


def test_directional_coverage_cohort_fails_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "_authority.json").write_text("{}", encoding="utf-8")
    (root / "_manifest.json").write_text("{}", encoding="utf-8")
    event_ids = [f"event-{index}" for index in range(245)]
    decision_ids = [f"decision-{index}" for index in range(245)]
    security_ids = [f"cik:{index % 169:010d}" for index in range(245)]
    attachments = pd.DataFrame(
        {
            "family_event_id": event_ids,
            "decision_id": decision_ids,
            "security_id": security_ids,
            "research_eligible": True,
            "identity_alignment": "exact_ticker_cik_compatible",
            "feature_available_at_utc": pd.Timestamp("2025-01-02T14:00:00Z"),
            "decision_time_utc": pd.Timestamp("2025-01-02T15:00:00Z"),
        }
    )
    decisions = pd.DataFrame(
        {
            "decision_id": decision_ids,
            "security_id": security_ids,
            "session_date_et": pd.date_range("2025-01-01", periods=245, freq="D").date,
            "development_fold": [index % 4 for index in range(245)],
            "validation_scope": [
                "unseen_security" if index < 41 else "seen_security"
                for index in range(245)
            ],
        }
    )
    monkeypatch.setattr(
        training,
        "load_intraday_event_preflight",
        lambda _directory, **_kwargs: SimpleNamespace(
            manifest={
                "status": "blocked",
                "training_eligible": False,
                "serving_eligible": False,
                "future_holdout_opened": False,
                "blockers": sorted(training.EXPECTED_PROXY_BLOCKERS),
                "event_authorities": [{"directory": "bound-parent"}],
            },
            attachments=attachments,
            decisions=decisions,
        ),
    )
    monkeypatch.setattr(
        training,
        "_load_event_subtypes",
        lambda _manifest, **_kwargs: pd.DataFrame(
            {"family_event_id": event_ids, "event_subtype": "coverage"}
        ),
    )

    with pytest.raises(DataReadinessError, match=r"coverage lacks governed capacity"):
        training.load_intraday_research_event_cohort(root, event_subtype="coverage")
