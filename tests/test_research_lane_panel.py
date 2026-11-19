"""Unit tests on synthetic files: sealed-period loading, identity mapping and the command surface."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from market_predictor.research_cli import app as research_app
from market_predictor.research_lane.panel import (
    PANEL_DIRECTORY,
    SealedPeriodError,
    check_development_end,
    load_research_panel,
)
from market_predictor.research_lane.run import score_price_predictors
from tests.support.research_lane_panel import POST_END_PRICE, bar_table, write_panel_files

END = date(2024, 5, 28)


@pytest.fixture()
def panel_root(tmp_path: Path) -> Path:
    root = tmp_path / "panel"
    write_panel_files(root, end=END)
    return root


def test_rows_after_development_end_never_reach_the_panel(panel_root: Path) -> None:
    panel = load_research_panel(panel_root, development_end=END)
    assert str(panel.sessions[-1]) == "2024-05-28"
    assert panel.sessions.max() <= np.datetime64(END, "D")
    for matrix in (panel.open, panel.high, panel.low, panel.close, panel.spy_close, panel.qqq_close):
        assert not np.any(np.abs(np.nan_to_num(matrix) - POST_END_PRICE) < 10.0)
    # year=2025 partition and the sector fund file are not parquet; reading either would have raised.
    assert panel.load_report["panel_years_opened"] == [2024]
    assert panel.load_report["sector_fund_files_not_read"] == ["XLK"]
    assert panel.development_end == END


def test_development_end_inside_sealed_period_is_refused_without_override(panel_root: Path) -> None:
    with pytest.raises(SealedPeriodError):
        check_development_end(date(2024, 5, 29))
    with pytest.raises(SealedPeriodError):
        load_research_panel(panel_root, development_end=date(2024, 6, 30))
    check_development_end(date(2024, 6, 30), unseal_final_test_period=True)


def test_ticker_change_and_ticker_reuse_map_to_stable_security_columns(tmp_path: Path) -> None:
    layout = write_panel_files(tmp_path / "panel", end=END)
    panel = load_research_panel(tmp_path / "panel", development_end=END)
    assert panel.security_ids == ("sec:A", "sec:B", "sec:C", "sec:D")
    report = panel.load_report
    assert report["securities_with_ticker_change"] == 1
    assert report["stock_bar_rows_without_membership_row"] == 0
    assert report["membership_rows_with_ticker_shared_by_two_securities"] == 0
    middle = np.datetime64(layout["middle"][0], "D")
    before, after = panel.sessions < middle, panel.sessions >= middle
    # sec:B spans OLD then NEW without a gap; sec:C and sec:D split the reused ticker XYZ.
    assert np.isfinite(panel.close[:, 1]).all()
    assert np.isfinite(panel.close[before, 2]).all() and np.isnan(panel.close[after, 2]).all()
    assert np.isnan(panel.close[before, 3]).all() and np.isfinite(panel.close[after, 3]).all()
    assert panel.eligible[:, 0].all()
    assert panel.sector_names == ("Energy", "Tech")
    assert (panel.sector[:, 1] == 0).all()


def _add_artifact(root: Path, ticker: str, file_ticker: str, days: list[date]) -> None:
    manifest_path = root / "combined_daily" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pq.write_table(bar_table(file_ticker, days, 5.0, END), root / "combined_daily" / "bars" / f"{ticker}.parquet")
    manifest["artifacts"].append({"ticker": ticker, "path": f"bars/{ticker}.parquet"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_bars_without_membership_rows_are_reported_not_placed(tmp_path: Path) -> None:
    root = tmp_path / "panel"
    layout = write_panel_files(root, end=END)
    _add_artifact(root, "ZZZ", "ZZZ", layout["days"])
    panel = load_research_panel(root, development_end=END)
    within = sum(1 for day in layout["days"] if day <= END)
    assert panel.load_report["stock_bar_rows_without_membership_row"] == within
    assert panel.load_report["stock_bar_rows_without_membership_row_by_ticker"]["ZZZ"]["rows"] == within
    assert "ZZZ" not in panel.security_ids and len(panel.security_ids) == 4


def test_bar_file_breaking_the_price_contract_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "panel"
    layout = write_panel_files(root, end=END)
    _add_artifact(root, "ZZZ", "OTHER", layout["days"])
    with pytest.raises(ValueError, match="not alpaca SIP"):
        load_research_panel(root, development_end=END)


def test_command_refuses_sealed_development_end_before_any_load(tmp_path: Path) -> None:
    result = CliRunner().invoke(research_app, ["score-price-predictors", "--root", str(tmp_path),
                                               "--development-end", "2024-06-03"])
    assert result.exit_code == 2
    assert "sealed period" in result.output + (result.stderr if result.stderr_bytes is not None else "")


def test_run_writes_scoreboard_and_trial_line_and_refuses_overwrite(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))  # isolated lease
    write_panel_files(tmp_path / PANEL_DIRECTORY, end=END)
    output = Path("data/research/research_lane/price_predictors_test")
    with patch("market_predictor.research_lane.run.guard"):
        result = score_price_predictors(root=tmp_path, output=output, development_end=END)
        assert result["output"] == output.as_posix()
        written = tmp_path / output
        board = json.loads((written / "scoreboard.json").read_text(encoding="utf-8"))
        assert "research label" in board["notice"]
        assert board["settings"]["development_end"] == "2024-05-28"
        assert "research label, adjusted prices" in (written / "scoreboard.md").read_text(encoding="utf-8")
        trials = (written.parent / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(trials) == 1 and json.loads(trials[0])["inputs"] == board["inputs"]
        with pytest.raises(FileExistsError):
            score_price_predictors(root=tmp_path, output=output, development_end=END)
