from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.edge_rebuild import intraday_event_training as training
from market_predictor.v3.errors import DataReadinessError


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
