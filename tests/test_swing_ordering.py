from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_predictor.edge_rebuild.swing_ordering import (
    audit_swing_ordering,
    load_complete_swing_ordering_audit,
)
from market_predictor.core.errors import DataReadinessError


def _policy(path: Path) -> None:
    path.write_text(
        """schema_version = "edge_rebuild.swing_ordering.v1"
score_features = ["signal_xs_rank"]
score_directions = [1]
top_quantile = 0.10
bottom_quantile = 0.10
minimum_scored_securities_per_session = 20
minimum_sessions = 60
minimum_mean_spread_bps = 5.0
minimum_positive_session_share = 0.50
minimum_newey_west_t_stat = 2.0
newey_west_lag_sessions = 10
""",
        encoding="utf-8",
    )


def _panel(*, reversed_outcome: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_number in range(60):
        session = pd.Timestamp("2024-01-02") + pd.offsets.BDay(session_number)
        decision = session.tz_localize("America/New_York").tz_convert("UTC")
        for security_number in range(20):
            signal = -1.0 + 2.0 * security_number / 19.0
            outcome = (-signal if reversed_outcome else signal) * 0.01
            rows.append(
                {
                    "security_id": f"sec:{security_number:02d}",
                    "session_date_et": session.date(),
                    "decision_time_utc": decision,
                    "barrier_label_available_at_utc": decision
                    + pd.Timedelta(days=10),
                    "feature_eligible": True,
                    "forward_return": outcome,
                    "signal_xs_rank": signal,
                }
            )
    return pd.DataFrame.from_records(rows)


def _arrange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reversed_outcome: bool,
) -> tuple[Path, Path, Path]:
    import market_predictor.edge_rebuild.swing_ordering as module

    panel = tmp_path / "panel"
    partition = panel / "final" / "panel" / "year=2024" / "part.parquet"
    partition.parent.mkdir(parents=True)
    _panel(reversed_outcome=reversed_outcome).to_parquet(partition, index=False)
    (panel / "final" / "_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_complete_swing_feature_panel",
        lambda _path: {
            "request_sha256": "panel-request",
            "files": [{"path": "panel/year=2024/part.parquet"}],
        },
    )
    policy = tmp_path / "policy.toml"
    _policy(policy)
    return panel, policy, tmp_path / "audit"


def test_ordering_gate_passes_a_monotonic_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, policy, output = _arrange(
        tmp_path,
        monkeypatch,
        reversed_outcome=False,
    )

    result = audit_swing_ordering(
        panel_dir=panel,
        config_path=policy,
        output_dir=output,
    )

    assert result["status"] == "passed"
    assert result["sessions"] == 60
    assert result["mean_session_spread_bps"] > 0.0
    assert result["newey_west_t_stat"] > 2.0
    assert (output / "_authority.json").is_file()
    assert (output / "session_spreads.parquet").is_file()
    assert load_complete_swing_ordering_audit(output)["status"] == "passed"


def test_ordering_gate_fails_a_reversed_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, policy, output = _arrange(
        tmp_path,
        monkeypatch,
        reversed_outcome=True,
    )

    result = audit_swing_ordering(
        panel_dir=panel,
        config_path=policy,
        output_dir=output,
    )

    assert result["status"] == "failed"
    assert result["mean_session_spread_bps"] < 0.0
    assert not result["gates"]["minimum_mean_spread_bps"]


def test_ordering_authority_rejects_a_mutated_session_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, policy, output = _arrange(
        tmp_path,
        monkeypatch,
        reversed_outcome=False,
    )
    audit_swing_ordering(
        panel_dir=panel,
        config_path=policy,
        output_dir=output,
    )
    with (output / "session_spreads.parquet").open("ab") as handle:
        handle.write(b"mutated")

    with pytest.raises(DataReadinessError, match="authority does not verify"):
        load_complete_swing_ordering_audit(output)
