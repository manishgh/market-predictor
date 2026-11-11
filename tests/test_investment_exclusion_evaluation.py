"""Pinned parent, stable exclusions and separate cohort publication."""
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import investment_exclusion_evaluation as module
from market_predictor.research.investment_selection_outcomes import _compare_cohorts

VIEW = "63/within_date_return_rank/shallow_boosted_return/held_out_stock"
A = "cik:0000000001"
B = "cik:0000000002"
C = "cik:0000000003"


def _write(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _holding(decision: str, session: str, security: str, ticker: str, stock: float | None,
             status: str = "reused_original") -> dict[str, Any]:
    return {
        "decision_id": decision, "horizon_sessions": 63, "decision_session": session, "security_id": security,
        "ticker": ticker, "status": status, "original_status": "source_validated" if status == "reused_original" else "unresolved",
        "stock_net_return": stock, "spy_gross_return": .05 if stock is not None else None,
        "net_excess_vs_spy": stock - .05 if stock is not None else None,
        "outcome_available_at_utc": "2026-11-11T01:00:00+00:00" if stock is not None else None,
        "verified_at_utc": "2026-11-11T00:30:00+00:00", "entry_time_utc": f"{session}T21:00:00+00:00",
        "economic_maturity_utc": "2024-08-01T20:00:00+00:00", "pair_proof": {"evidence": decision} if status == "recovered" else None,
    }


@pytest.fixture
def parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(module, "guard", lambda *_args: None)
    monkeypatch.setattr("market_predictor.research.investment_features.guard", lambda *_args: None)
    directory = tmp_path / "parent"
    directory.mkdir()
    rows = [
        _holding("jan-b", "2024-01-31", B, "B", .118),
        _holding("jan-a", "2024-01-31", A, "A", .998),
        _holding("jan-c", "2024-01-31", C, "C", .018, "recovered"),
        _holding("feb-a", "2024-02-29", A, "OLD", None, "unresolved"),
        _holding("feb-b", "2024-02-29", B, "B", .218),
        _holding("mar-a", "2024-03-28", A, "A", .318),
        _holding("apr-c", "2024-04-30", C, "C", None, "unresolved"),
        _holding("apr-b", "2024-04-30", B, "B", .018),
        _holding("may-b", "2024-05-31", B, "B", .078),
    ]
    keys = {(row["decision_id"], 63): row for row in rows}
    cohorts = []
    for session in sorted({row["decision_session"] for row in rows}):
        members = [row for row in rows if row["decision_session"] == session]
        cohorts.append({"decision_session": session, "security_ids": [row["security_id"] for row in members],
                        "keys": [(row["decision_id"], 63) for row in members]})
    selection = {VIEW: [{name: entry[name] for name in ("decision_session", "security_ids")} for entry in cohorts]}
    _write(directory / "outcomes.json", {"rows": rows})
    _write(directory / "comparisons.json", {"views": _compare_cohorts({VIEW: cohorts}, keys)})
    manifest = directory / "_manifest.json"
    pin = _write(manifest, {
        "schema": "market_predictor.investment_selection_outcomes", "scope": "retrospective_research",
        "status": "complete_with_explicit_gaps", "selection_sha256": json_sha256(selection),
        "files": {name: file_sha256(directory / name) for name in ("outcomes.json", "comparisons.json")},
        "round_trip_stock_cost_bps": 20, "benchmark_basis": "stock_net_minus_spy_gross",
        "selection_changed": False, "models_refitted": False, "original_labels_changed": False, **module.FLAGS,
        "unique_selected_keys": 9, "original_unknown_keys": 3, "reused_original_keys": 6, "recovered_keys": 1,
        "unresolved_keys": 2, "comparison_views": 1, "comparison_cohorts": 5, "selected_positions": 9,
    })
    request = tmp_path / "request.json"
    _write(request, {"schema_version": "market_predictor.investment_exclusion_evaluation_request",
        "scope": "post_observation_sensitivity", "parent_manifest": pin,
        "exclusions": [{"ticker": "A", "security_id": A, "reason": "user_requested_exclusion"}]})
    return request, manifest


def _run(root: Path, request: Path) -> dict[str, Any]:
    return module.evaluate_investment_exclusions(root=root, request=request, request_sha256=file_sha256(request),
        output=root / "sensitivity")


def _repin(request: Path, manifest: Path, name: str | None = None) -> None:
    report = _read(manifest)
    if name is not None:
        report["files"][name] = file_sha256(manifest.parent / name)
    pin = _write(manifest, report)
    value = _read(request)
    value["parent_manifest"] = pin
    _write(request, value)


def test_exclusions_preserve_order_status_clock_and_cost_without_replacement(
    tmp_path: Path, parent: tuple[Path, Path],
) -> None:
    request, manifest = parent
    before = {path: file_sha256(path) for path in manifest.parent.iterdir()}
    parent_rows = _read(manifest.parent / "outcomes.json")["rows"]
    old_view = _read(manifest.parent / "comparisons.json")["views"][VIEW]
    result = _run(tmp_path, request)
    output = tmp_path / "sensitivity"
    report = _read(output / "_manifest.json")
    assert result["manifest_sha256"] == file_sha256(output / "_manifest.json")
    assert all(file_sha256(output / name) == digest for name, digest in report["files"].items())
    assert report["parent_manifest"]["sha256"] == before[manifest]
    assert report["parent_comparisons"]["sha256"] == before[manifest.parent / "comparisons.json"]
    assert report["source_files"][str(request)] == file_sha256(request)
    assert report["outcome_verification_basis"] == "pinned_parent_outcomes"
    assert report["status"] == "complete_with_explicit_gaps"
    assert report["round_trip_stock_cost_bps"] == 20
    assert all(report[flag] is False for flag in (*module.FLAGS, "reranked", "replacements_added", "training_ready"))
    assert _read(output / "outcomes.json")["rows"] == [row for row in parent_rows if row["security_id"] != A]
    assert report["coverage"]["removed"] == {"unique_keys": 3, "selected_positions": 3,
        "key_status_counts": {"reused_original": 2, "recovered": 0, "unresolved": 1},
        "position_status_counts": {"reused_original": 2, "recovered": 0, "unresolved": 1}}
    assert report["exclusions"][0]["removed_keys"] == report["coverage"]["removed"]["key_status_counts"]
    view = _read(output / "comparisons.json")["views"][VIEW]
    assert (view["date_count"], view["complete_dates"], view["empty_dates"]) == (5, 3, 1)
    assert view["mean_net_excess_vs_spy"] is None
    assert (view["original_selected_positions"], view["selected_positions"], view["removed_positions"]) == (9, 6, 3)
    january, february, march, april, may = view["dates"]
    assert january["decision_ids"] == ["jan-b", "jan-c"]
    assert january["security_ids"] == [B, C]
    assert (january["original_selected_positions"], january["selected_positions"]) == (3, 2)
    assert january["mean_stock_net_return"] == pytest.approx(.068)
    assert january["mean_spy_gross_return"] == pytest.approx(.05)
    assert january["mean_net_excess_vs_spy"] == pytest.approx(.018)
    assert february["complete"] and not february["original_complete"]
    assert february["removed_positions"] == [{"decision_id": "feb-a", "security_id": A, "ticker": "OLD",
        "horizon_sessions": 63, "status": "unresolved", "reason": "user_requested_exclusion"}]
    assert february["mean_net_excess_vs_spy"] == pytest.approx(.168)
    assert march["selected_positions"] == 0 and not march["complete"] and march["mean_stock_net_return"] is None
    assert april["decision_ids"] == ["apr-c", "apr-b"] and april["unresolved_positions"] == 1
    assert not april["complete"] and april["mean_stock_net_return"] is None
    assert {name: may[name] for name in old_view["dates"][-1]} == old_view["dates"][-1]
    with pytest.raises(FileExistsError):
        _run(tmp_path, request)
    assert {path: file_sha256(path) for path in before} == before


def test_shared_math_leaves_empty_views_unavailable() -> None:
    result = _compare_cohorts({VIEW: []}, {})[VIEW]
    assert result["date_count"] == result["complete_dates"] == 0
    assert result["mean_net_excess_vs_spy"] is None


def test_shared_math_equal_weights_complete_dates() -> None:
    jan_a = _holding("a", "2024-01-31", A, "A", .098)
    jan_b = _holding("b", "2024-01-31", B, "B", .198)
    feb_b = _holding("c", "2024-02-29", B, "B", .398)
    holdings = {(row["decision_id"], 63): row for row in (jan_a, jan_b, feb_b)}
    cohorts = [
        {"decision_session": "2024-01-31", "keys": [("a", 63), ("b", 63)], "security_ids": [A, B]},
        {"decision_session": "2024-02-29", "keys": [("c", 63)], "security_ids": [B]},
    ]
    result = _compare_cohorts({VIEW: cohorts}, holdings)[VIEW]
    assert result["mean_stock_net_return"] == pytest.approx((.148 + .398) / 2)
    assert result["mean_net_excess_vs_spy"] == pytest.approx((.098 + .348) / 2)


@pytest.mark.parametrize(("poison", "message"), [
    ("parent_hash", "hash differs"),
    ("sidecar_hash", "hash differs"),
    ("sidecar_path", "closed fixed-selection parent"),
    ("parent_path", "escapes repository"),
    ("parent_count", "manifest counts"),
    ("cohort_count", "cohort counts or means"),
    ("cohort_mean", "cohort counts or means"),
    ("cohort_order", "ordered selection hash"),
    ("cohort_duplicate", "duplicates"),
    ("cohort_horizon", "placement does not bind"),
    ("cohort_security", "placement does not bind"),
    ("cohort_date", "placement does not bind"),
    ("outcome_duplicate", "duplicate a decision"),
    ("unresolved_return", "unresolved parent holding"),
    ("invalid_return", "returns are invalid"),
    ("ambiguous_ticker", "does not uniquely bind"),
    ("wrong_id", "does not uniquely bind"),
    ("duplicate_exclusion", "must be unique"),
])
def test_bad_parent_or_exclusion_cannot_publish(tmp_path: Path, parent: tuple[Path, Path], poison: str, message: str) -> None:
    request, manifest = parent
    comparisons = manifest.parent / "comparisons.json"
    outcomes = manifest.parent / "outcomes.json"
    if poison == "parent_hash":
        manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    elif poison == "sidecar_hash":
        outcomes.write_text(outcomes.read_text() + "\n", encoding="utf-8")
    elif poison == "sidecar_path":
        report = _read(manifest)
        report["files"]["../outcomes.json"] = report["files"].pop("outcomes.json")
        _write(manifest, report)
        _repin(request, manifest)
    elif poison == "parent_path":
        value = _read(request)
        value["parent_manifest"]["path"] = "../outside/_manifest.json"
        _write(request, value)
    elif poison == "parent_count":
        report = _read(manifest)
        report["unique_selected_keys"] += 1
        _write(manifest, report)
        _repin(request, manifest)
    elif poison.startswith("cohort_"):
        value = _read(comparisons)
        first = value["views"][VIEW]["dates"][0]
        if poison == "cohort_order":
            first["decision_ids"].reverse()
            first["security_ids"].reverse()
        elif poison == "cohort_duplicate":
            first["decision_ids"][1] = first["decision_ids"][0]
        elif poison == "cohort_count":
            first["selected_positions"] += 1
        elif poison == "cohort_mean":
            first["mean_stock_net_return"] = 999.0
        elif poison == "cohort_horizon":
            value["views"]["252" + VIEW[2:]] = value["views"].pop(VIEW)
        elif poison == "cohort_security":
            first["security_ids"][0] = "cik:0000000099"
        else:
            first["decision_session"] = "2024-01-30"
        _write(comparisons, value)
        _repin(request, manifest, comparisons.name)
    elif poison in {"outcome_duplicate", "unresolved_return", "invalid_return", "ambiguous_ticker"}:
        value = _read(outcomes)
        if poison == "outcome_duplicate":
            value["rows"].append(value["rows"][0])
        elif poison == "unresolved_return":
            value["rows"][3]["stock_net_return"] = .1
        elif poison == "invalid_return":
            value["rows"][0]["net_excess_vs_spy"] = .1
        else:
            value["rows"][0]["ticker"] = "A"
        _write(outcomes, value)
        _repin(request, manifest, outcomes.name)
    else:
        value = _read(request)
        if poison == "wrong_id":
            value["exclusions"][0]["security_id"] = C
        else:
            value["exclusions"].append(value["exclusions"][0])
        _write(request, value)
    with pytest.raises((DataReadinessError, ValueError), match=message):
        _run(tmp_path, request)
    assert not (tmp_path / "sensitivity").exists()



def test_parent_change_during_staging_prevents_publication(
    tmp_path: Path, parent: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, manifest = parent
    write = module.write_json_object

    def change_parent(path: Path, value: dict[str, Any]) -> None:
        write(path, value)
        if path.name == "comparisons.json":
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "write_json_object", change_parent)
    with pytest.raises(DataReadinessError, match="source changed"):
        _run(tmp_path, request)
    assert not (tmp_path / "sensitivity").exists()


def test_cli_forwards_pinned_exclusion_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "evaluate_investment_exclusions",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    result = CliRunner().invoke(app, ["evaluate-investment-exclusions", "--root", str(tmp_path),
        "--request", "request.json", "--request-sha256", "a" * 64, "--output", "separate"])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": Path("request.json"), "request_sha256": "a" * 64, "output": Path("separate")}]

