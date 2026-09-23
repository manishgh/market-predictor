"""Cohort inventory over archives written by the real collector, derivation and verifiers."""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import issuer_content_cohort_inventory as cohort
from market_predictor.swing.contracts.research_cohort import ResearchSecurityExclusion, SwingResearchCohort
from market_predictor.swing.datasets.issuer_news_preparation import _source
from market_predictor.universe.issuer_news_identity import map_news_coverage, map_news_relations
from tests.support.alpaca_news_archive import article, collect, corrected_spec, derive, saved_authority

AAA, BBB, CCC, DDD, EEE, FFF = (("AAA", "cik:0000000001:ticker:AAA"), ("BBB", "cik:0000000002"),
    ("CCC", "sp500-historical:ccc"), ("DDD", "cik:0000000004:ticker:DDD"), ("EEE", "cik:0000000005:ticker:EEE"),
    ("FFF", "cik:0000000006"))
COHORT = ("cik:0000000001", "cik:0000000002", "cik:0000000005", "cik:0000000006", "cik:0000000007")


def _early_news(symbol: str, start: datetime, _end: datetime) -> list[list[dict[str, Any]]]:
    if start.year != 2019:
        return [[article(250, "2020-02-01T15:00:00Z", symbols=["BBB"], content="", summary="")]] if symbol == "BBB" else []
    items = {"AAA": [[article(101, "2019-08-01T15:00:00Z", symbols=["AAA"]),
                      article(103, "2019-07-09T03:00:00Z", symbols=["AAA"], updated="2019-07-09T05:00:00Z"),
                      article(199, "2019-08-02T15:00:00Z", symbols=["MSFT"])],
                     [article(102, "2020-01-01T04:30:00Z", symbols=["AAA"], content="")]],
             "BBB": [[article(201, "2019-09-01T15:00:00Z", symbols=["BBB"])]],
             "CCC": [[article(301, "2019-09-02T15:00:00Z", symbols=["CCC"])]],
             "DDD": [[article(401, "2019-09-03T15:00:00Z", symbols=["DDD"])]],
             "EEE": [[article(501, "2019-09-04T15:00:00Z", symbols=["EEE"])]]}
    return items.get(symbol, [])


def _later_news(symbol: str, _start: datetime, _end: datetime) -> list[list[dict[str, Any]]]:
    if symbol != "AAA":
        return []
    return [[article(601, "2024-01-10T15:00:00Z", symbols=["AAA"]),
             article(602, "2024-05-20T15:00:00Z", symbols=["AAA"], updated="2024-06-05T15:00:00Z"),
             article(603, "2024-06-10T15:00:00Z", symbols=["AAA"])]]


def _corrected_news(symbol: str, start: datetime, _end: datetime) -> list[list[dict[str, Any]]]:
    if start.month == 7:
        return [[article(701, "2019-08-01T16:00:00Z", symbols=[symbol])]]
    return []


def _bridge(root: Path) -> dict[str, str]:
    bridge = pd.DataFrame([{"source_security_id": source, "ticker": ticker, "target_security_id": source.split(":ticker:")[0],
        "effective_from_utc": pd.Timestamp("2019-07-09T04:00:00Z"), "effective_to_utc": pd.Timestamp("2026-07-09T04:00:00Z"),
        "available_at_utc": pd.Timestamp("2019-07-09T04:00:00Z"), "bridge_row_sha256": f"{index:064x}"}
        for index, (ticker, source) in enumerate((AAA, DDD, EEE), start=1)])
    path = root / "data/research/identity/identity_bridge.parquet"
    write_canonical_artifact(bridge, path, artifact_type="issuer_news_identity_bridge", production_ready=False,
        audit=CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic", status="pass", failures=0,
                                                               rows_checked=len(bridge), detail="test-only bridge"),)))
    files = {item.relative_to(root).as_posix(): file_sha256(item) for item in (path, manifest_path_for(path))}
    manifest = root / "data/research/identity/_manifest.json"
    manifest.write_text(json.dumps({"schema": "market_predictor.issuer_news_identity_alignment", "source_files": files}),
                        encoding="utf-8")
    return {"path": manifest.relative_to(root).as_posix(), "sha256": file_sha256(manifest)}


def _population(root: Path) -> Path:
    """A valid approved-population audit for the real cohort loader, with one pinned source file."""
    source = root / "data/reports/population_source.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("test-only cohort source", encoding="utf-8")
    model = SwingResearchCohort(schema_version="market_predictor.swing_research_cohort",
        scope="retrospective_development_restriction", price_basis_status="not_certified_by_cohort",
        combined_daily_inputs_sha256="e" * 64, original_security_ids=tuple(sorted((*COHORT, "cik:0000000008"))),
        inherited_excluded_security_ids=(), warmup_only_security_ids=(),
        exclusions=(ResearchSecurityExclusion(security_id="cik:0000000008", tickers=("HHH",), reason="unavailable_trading"),),
        maximum_exclusion_bps=5000, cap_approval_reference="test-only",
        source_files={"data/reports/population_source.txt": file_sha256(source)})
    payload = {"cohort": model.model_dump(mode="json"), "cohort_sha256": model.sha256(), "summary": model.summary(),
               "coverage": {"test_only": True}}
    path = root / "data/reports/population.json"
    path.write_text(json.dumps({**payload, "audit_sha256": json_sha256(payload)}), encoding="utf-8")
    return path


def _world(root: Path, monkeypatch: pytest.MonkeyPatch, *, corrected: tuple[tuple[str, str], ...] = (FFF,)) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    early = collect(root, monkeypatch, "early_news", [AAA, BBB, CCC, DDD, EEE], _early_news,
                    start=date(2019, 7, 9), end=date(2020, 7, 8), chunk_days=183)
    later = collect(root, monkeypatch, "later_news", [AAA, BBB], _later_news,
                    start=date(2023, 7, 11), end=date(2024, 7, 10), chunk_days=366)
    fixed = collect(root, monkeypatch, "corrected_news", list(corrected), _corrected_news,
                    start=date(2019, 7, 9), end=date(2019, 12, 31), chunk_days=92)
    sources = {"early": derive(root, saved_authority(root, early, "early_saved", blindspots=(EEE[1],)), "early_derived"),
        "later": derive(root, saved_authority(root, later, "later_saved"), "later_derived"),
        "corrected": corrected_spec(root, fixed, saved_authority(root, fixed, "corrected_saved", corrected=True))}
    monthly = root / "configs/monthly.json"
    monthly.parent.mkdir(parents=True, exist_ok=True)
    monthly.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    population = _population(root)
    config = root / "configs/cohort.json"
    config.write_text(json.dumps({"schema": cohort.CONFIG_SCHEMA,
        "monthly_news_config": {"path": "configs/monthly.json", "sha256": file_sha256(monthly)},
        "identity_manifest": _bridge(root),
        "approved_population": {"path": "data/reports/population.json", "sha256": file_sha256(population)}}), encoding="utf-8")
    return {"root": root, "config": config, "config_sha256": file_sha256(config)}


@pytest.fixture(autouse=True)
def _test_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohort, "_guard", lambda: None)
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        return _world(tmp_path_factory.mktemp("cohort") / "repo", patch)


def _run(world: dict[str, Any], name: str, **changes: Any) -> dict[str, Any]:
    return cohort.publish_cohort_content_inventory(root=world["root"], config=world["config"],
        config_sha256=world["config_sha256"], output=world["root"] / "data/research" / name, **changes)


def _frames(world: dict[str, Any], name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = world["root"] / "data/research" / name
    records = pd.concat([pd.read_parquet(path) for path in sorted((output / "parts").glob("*.parquet"))], ignore_index=True)
    return (records.sort_values(["archive", "chunk_id", "event_id"]).reset_index(drop=True),
            pd.read_parquet(output / "coverage.parquet"), pd.read_parquet(output / "security_years.parquet"))


def test_statuses_attribution_coverage_and_years(world: dict[str, Any]) -> None:
    report = _run(world, "complete")
    totals = report["totals"]
    assert totals["units"] == {"corrected/observed/artifact_pinned_sidecar_observed": 1,
        "corrected/observed_empty/pages_verified_empty": 1, "early/observed/derivation_unavailable_raw_pinned": 1,
        "early/observed/sidecar_pinned": 5, "early/observed_empty/pages_verified_empty": 4,
        "later/observed/sidecar_pinned": 1, "later/observed_empty/pages_verified_empty": 1}
    assert totals["derivation_unavailable"] == {"old_attribution_or_sentiment_not_published": 1}
    assert totals["cohort_securities_with_proven_query"] == 4 and totals["cohort_securities_without_proven_query"] == 1
    # Story 103 is attributed at its availability while its publication precedes the bridge span.
    assert totals["bridged_included_records_outside_their_coverage_segment"] == 1
    assert {key.split("/")[1] for key in totals["unattributed_records"]} == {"bridged_non_cohort", "no_proven_identity"}
    # CCC and DDD never resolve; AAA, DDD and EEE queries also start before their bridge span opens.
    assert totals["unattributed_query_ids"] == {"bridged_non_cohort": 1, "no_proven_identity": 1, "outside_bridge_span": 3}
    assert sum(totals["unattributed_coverage_days"].values()) > 0 and set(report["evidence_levels"]) == set(cohort._EVIDENCE)
    assert report["known_empty_scope"] == cohort.KNOWN_EMPTY_SCOPE
    for flag in ("training_eligible", "serving_eligible", "promotion_eligible"):
        assert report[flag] is False
    assert report["attribution_status"] == report["content_qualification"] == "not_established"
    records, coverage, years = _frames(world, "complete")
    by_story = dict(zip(records.provider_story_id, records.query_identity_resolution, strict=True))
    assert by_story == {"101": "bridged", "102": "bridged", "103": "bridged", "201": "identity_equal", "250": "identity_equal",
        "301": "no_proven_identity", "401": "bridged_non_cohort", "501": "bridged", "601": "bridged", "602": "bridged",
        "701": "identity_equal"}
    assert records.loc[records.provider_story_id.eq("102"), "publication_year_new_york"].item() == 2019
    assert records.loc[records.provider_story_id.eq("602"), "inventory_status"].item() == "version_after_cutoff"
    assert "603" not in set(records.provider_story_id)
    locators = [entry["page_path"] for value in records.raw_record_locators_json for entry in json.loads(value)]
    assert locators and not any(Path(path).is_absolute() or "\\" in path for path in locators)
    assert set(coverage.query_identity_resolution) == {"bridged", "outside_bridge_span", "identity_equal",
                                                       "bridged_non_cohort", "no_proven_identity"}
    assert coverage.query_provider_symbol.notna().all() and "unavailable_saved_evidence" in set(coverage.derivation_status)
    summaries = [unit for path in (world["root"] / "data/research/complete/parts").glob("*.units.json")
                 for unit in json.loads(path.read_text())["units"]]
    assert {unit["known_empty_scope"] for unit in summaries if unit["status"] == "observed_empty"} == {cohort.KNOWN_EMPTY_SCOPE}
    aaa = years.set_index(["cohort_security_id", "year_new_york"]).loc["cik:0000000001"]
    assert (aaa.at[2019, "included_stories"], aaa.at[2024, "included_stories"], aaa.at[2024, "version_after_cutoff_stories"]) == (3, 1, 1)
    assert aaa.at[2019, "unknown_no_proven_query_days"] == pytest.approx(4 / 24)
    assert aaa.at[2019, "covered_sidecar_pinned_days"] + aaa.at[2019, "unknown_no_proven_query_days"] == pytest.approx(
        aaa.at[2019, "window_days"])
    assert aaa.at[2020, "covered_pages_verified_empty_days"] > 0
    unreached = years.loc[years.cohort_security_id.eq("cik:0000000007")]
    assert (unreached.unknown_no_proven_query_days == unreached.window_days).all() and unreached.query_returned_stories.eq(0).all()
    eee = years.loc[years.cohort_security_id.eq("cik:0000000005")]
    assert eee.covered_derivation_unavailable_raw_pinned_days.sum() > 0 and eee.covered_artifact_pinned_sidecar_observed_days.eq(0).all()
    assert set(years.attribution_status) == {"not_established"} and len(years) == len(COHORT) * 6
    with pytest.raises(DataReadinessError, match="immutable"):
        _run(world, "complete")


def test_part_size_and_ledger_order_do_not_change_results(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _run(world, "one-part-size")
    monkeypatch.setattr(cohort, "PART_CHUNKS", 1)
    monkeypatch.setattr(cohort, "_source", lambda root, spec: {**(value := _source(root, spec)),
                                                              "ledger": value["ledger"].iloc[::-1].reset_index(drop=True)})
    _run(world, "other-part-size")
    for left, right in zip(_frames(world, "one-part-size"), _frames(world, "other-part-size"), strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_resume_requires_pinned_checkpoint_and_verified_parts(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohort, "PART_CHUNKS", 2)
    inspect = cohort._inspect
    calls: list[None] = []

    def interrupted(*args: Any) -> Any:
        calls.append(None)
        if len(calls) == 5:
            raise OSError("test-only interruption")
        return inspect(*args)

    monkeypatch.setattr(cohort, "_inspect", interrupted)
    with pytest.raises(OSError, match="interruption"):
        _run(world, "resumed")
    monkeypatch.setattr(cohort, "_inspect", inspect)
    output = world["root"] / "data/research/resumed"
    checkpoint = file_sha256(output / "_checkpoint.json")
    assert json.loads((output / "_checkpoint.json").read_text())["parts"]
    with pytest.raises(DataReadinessError, match="checkpoint SHA256"):
        _run(world, "resumed")
    with pytest.raises(DataReadinessError, match="pin differs"):
        _run(world, "resumed", resume_checkpoint_sha256="0" * 64)
    part = next((output / "parts").glob("*.parquet"))
    original = part.read_bytes()
    part.write_bytes(original + b"tamper")
    with pytest.raises(DataReadinessError, match="checkpointed part changed"):
        _run(world, "resumed", resume_checkpoint_sha256=checkpoint)
    part.write_bytes(original)
    _run(world, "resumed", resume_checkpoint_sha256=checkpoint)
    _run(world, "uninterrupted")
    for left, right in zip(_frames(world, "resumed"), _frames(world, "uninterrupted"), strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_ledger_parity_rejects_every_producer_statistic(world: dict[str, Any]) -> None:
    captured: list[tuple[dict[str, Any], dict[str, Any]]] = []
    parity = cohort._parity

    def record(unit: dict[str, Any], summary: dict[str, Any]) -> None:
        captured.append((unit, summary))
        parity(unit, summary)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cohort, "_parity", record)
        _run(world, "parity-capture")
    unit, summary = next((u, s) for u, s in captured if u["archive"] == "later" and u["status"] == "observed")
    assert summary["canonical_rows"] == 3 and summary["included_rows"] == unit["included_rows"] == 1
    for key, value in (("query_chunk_pages", 2), ("query_chunk_provider_records", 9), ("query_chunk_admitted_stories", 9),
                       ("canonical_rows", 4), ("included_rows", 3), ("query_window_end_exclusive_utc", "2024-06-01T00:00:00+00:00")):
        with pytest.raises(DataReadinessError, match="differ"):
            cohort._parity(unit, {**summary, key: value})
    for reason in ("clock", "title", "window", "symbol"):
        discarded = {**summary["query_chunk_discarded_records"], reason: 7}
        with pytest.raises(DataReadinessError, match="producer statistics"):
            cohort._parity(unit, {**summary, "query_chunk_discarded_records": discarded})


@pytest.mark.parametrize("field,value", [("config_sha256", "0" * 64), ("config", "extra_key"), ("config", "bare_pin")])
def test_config_is_pinned_and_exact(world: dict[str, Any], field: str, value: str) -> None:
    changes: dict[str, Any] = {"config_sha256": value} if field == "config_sha256" else {}
    if field == "config":
        path = world["root"] / f"configs/{value}.json"
        settings = json.loads(world["config"].read_text())
        changed = {**settings, "extra_key": True} if value == "extra_key" else {
            **settings, "identity_manifest": settings["identity_manifest"]["path"]}
        path.write_text(json.dumps(changed), encoding="utf-8")
        changes = {"config": path, "config_sha256": file_sha256(path)}
    with pytest.raises(DataReadinessError):
        cohort.publish_cohort_content_inventory(**{"root": world["root"], "config": world["config"],
            "config_sha256": world["config_sha256"], "output": world["root"] / "data/research" / f"bad-{value}", **changes})
    assert not (world["root"] / "data/research" / f"bad-{value}").exists()


def test_busy_lease_prevents_any_source_read(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohort, "_source", lambda *_: pytest.fail("source read without the lease"))
    with heavy_job_lease("test-owner", runtime_dir=world["root"] / "data/runtime"), pytest.raises(HeavyJobBusyError):
        _run(world, "busy")
    assert not (world["root"] / "data/research/busy").exists()


def test_derived_sources_require_their_declared_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = _world(tmp_path / "repo", monkeypatch)
    moved = tmp_path / "moved"
    shutil.copytree(built["root"], moved)
    # The reused derivation loader rejects moved children first; the declared-root check covers parent roots.
    with pytest.raises(DataReadinessError, match="escapes its authority|declared repository root"):
        cohort.publish_cohort_content_inventory(root=moved, config=moved / "configs/cohort.json",
            config_sha256=built["config_sha256"], output=moved / "data/research/moved")


def test_unpinned_sidecar_must_bind_artifact_and_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = _world(tmp_path / "repo", monkeypatch)
    sidecar = next((built["root"] / "data/raw/corrected_news/events").glob("*.parquet.manifest.json"))
    value = json.loads(sidecar.read_text())
    value["inputs"]["collection_request_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="does not bind"):
        _run(built, "unbound")


def test_overlapping_attributed_coverage_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = _world(tmp_path / "repo", monkeypatch, corrected=(FFF, BBB))
    with pytest.raises(DataReadinessError, match="overlaps"):
        _run(built, "overlap")


@pytest.mark.parametrize("target", ["configs/monthly.json", "data/research/identity/_manifest.json",
    "data/research/identity/identity_bridge.parquet", "data/research/identity/identity_bridge.parquet.manifest.json",
    "data/reports/population.json", "data/reports/population_source.txt", "data/research/early_derived/_manifest.json",
    "data/research/early_derived/_source_children.json", "data/raw/early_news/_manifest.json",
    "data/research/corrected_saved/audit.json"])
def test_every_input_pin_is_verified_before_any_part(world: dict[str, Any], target: str) -> None:
    path, output = world["root"] / target, world["root"] / "data/research" / f"tamper-{Path(target).name}"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(DataReadinessError):
            cohort.publish_cohort_content_inventory(root=world["root"], config=world["config"],
                config_sha256=world["config_sha256"], output=output)
    finally:
        path.write_bytes(original)
    assert not (output / "parts").exists()


def test_resume_rejects_changed_request_checkpoint_parts_and_unbound_files(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cohort, "PART_CHUNKS", 2)
    inspect = cohort._inspect
    calls: list[None] = []

    def interrupted(*args: Any) -> Any:
        calls.append(None)
        if len(calls) == 5:
            raise OSError("test-only interruption")
        return inspect(*args)

    monkeypatch.setattr(cohort, "_inspect", interrupted)
    with pytest.raises(OSError, match="interruption"):
        _run(world, "resume-matrix")
    monkeypatch.setattr(cohort, "_inspect", inspect)
    output = world["root"] / "data/research/resume-matrix"
    checkpoint = output / "_checkpoint.json"
    original, pin = checkpoint.read_bytes(), file_sha256(checkpoint)
    monkeypatch.setattr(cohort, "PART_CHUNKS", 3)
    with pytest.raises(DataReadinessError, match="request differs"):
        _run(world, "resume-matrix", resume_checkpoint_sha256=pin)
    monkeypatch.setattr(cohort, "PART_CHUNKS", 2)
    value = json.loads(original)
    checkpoint.write_text(json.dumps({**value, "request_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="checkpoint request differs"):
        _run(world, "resume-matrix", resume_checkpoint_sha256=file_sha256(checkpoint))
    units = output / "parts/early-0000.units.json"
    saved = units.read_bytes()
    payload = json.loads(saved)
    units.write_text(json.dumps({**payload, "units": payload["units"][::-1]}), encoding="utf-8")
    value["parts"]["early-0000"]["units_sha256"] = file_sha256(units)
    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="covers other chunks"):
        _run(world, "resume-matrix", resume_checkpoint_sha256=file_sha256(checkpoint))
    units.write_bytes(saved)
    checkpoint.write_bytes(original)
    (output / "parts/foreign.units.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="differ from its checkpoint"):
        _run(world, "resume-matrix", resume_checkpoint_sha256=pin)
    stray = world["root"] / "data/research/stray"
    stray.mkdir()
    (stray / "note.txt").write_text("unrelated", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="without a bound request"):
        _run(world, "stray")


def test_crash_in_the_first_part_leaves_a_resumable_empty_checkpoint(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = cohort._inspect
    monkeypatch.setattr(cohort, "_inspect", Mock(side_effect=OSError("test-only first unit failure")))
    with pytest.raises(OSError, match="first unit"):
        _run(world, "first-crash")
    monkeypatch.setattr(cohort, "_inspect", inspect)
    checkpoint = world["root"] / "data/research/first-crash/_checkpoint.json"
    assert json.loads(checkpoint.read_text())["parts"] == {}
    report = _run(world, "first-crash", resume_checkpoint_sha256=file_sha256(checkpoint))
    assert report["status"] == "complete_inventory_only"


@pytest.mark.parametrize("case,message", [("root", "declared repository root"), ("escape", "escapes its declared repository"),
    ("pin", "derived child and raw artifact pins differ"), ("reason", "lacks its reason"),
    ("ledger", "lacks one original coverage ledger"), ("duplicate", "unavailable chunks are duplicated")])
def test_derived_unit_resolution_fails_closed(world: dict[str, Any], case: str, message: str) -> None:
    root = world["root"]
    copy = root / "data/research" / f"early-{case}"
    shutil.copytree(root / "data/research/early_derived", copy)
    manifest = json.loads((copy / "_manifest.json").read_text())
    children_path = copy / "_source_children.json"
    children = json.loads(children_path.read_text())
    if case == "root":
        manifest["source_path_resolution"]["repository_root"] = str(root.parent)
    elif case == "escape":
        children[str(root.parent / "outside.parquet")] = "0" * 64
    elif case == "pin":
        key = next(key for key in children if "/data/raw/early_news/events/" in key.replace("\\", "/") and key.endswith(".parquet"))
        children[key] = "0" * 64
    elif case == "reason":
        manifest["unavailable_chunks"] = []
    elif case == "duplicate":
        manifest["unavailable_chunks"] = manifest["unavailable_chunks"] * 2
    else:
        children = {key: value for key, value in children.items() if not key.endswith("_source_collections.parquet")}
    children_path.write_text(json.dumps(children), encoding="utf-8")
    manifest["inventory"]["_source_children.json"] = file_sha256(children_path)
    (copy / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ledger, _ = load_canonical_artifact(copy / "collection/source_collections.parquet", expected_type="source_collections",
                                        allow_research=True)
    spec = {"kind": "derived", "directory": copy.relative_to(root).as_posix(), "manifest_sha256": file_sha256(copy / "_manifest.json")}
    with pytest.raises(DataReadinessError, match=message):
        cohort._derived_units(root, "early", spec, {"ledger": ledger}, {})


def test_identity_mapping_order_is_verified(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohort, "map_news_relations", lambda frame, bridge: map_news_relations(frame, bridge).iloc[::-1])
    with pytest.raises(DataReadinessError, match="reordered"):
        _run(world, "reordered")


def test_story_precedence_prefers_included_then_archive_order() -> None:
    columns = {name: ["x", "x"] for name in cohort._SUMMARY_COLUMNS}
    records = pd.DataFrame({**columns, "archive": ["early", "later"], "chunk_id": ["a", "b"], "event_id": ["e1", "e2"],
        "cohort_security_id": ["cik:0000000001"] * 2, "source_family": ["alpaca"] * 2, "provider_story_id": ["9", "9"],
        "inventory_status": ["version_after_cutoff", "included"], "content_category": ["headline_only", "provider_body_field"],
        "published_at_utc": pd.to_datetime(["2020-03-01T15:00:00Z", "2021-03-01T15:00:00Z"], utc=True),
        "publication_year_new_york": [2020, 2021]})
    coverage = pd.DataFrame({"cohort_security_id": ["cik:0000000001"], "evidence": ["sidecar_pinned"], "ticker": ["AAA"],
        "requested_start_utc": pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True),
        "requested_end_utc": pd.to_datetime(["2021-06-01T00:00:00Z"], utc=True)})
    years = cohort._security_years(records, coverage, ("cik:0000000001",)).set_index("year_new_york")
    assert (years.at[2021, "included_stories"], years.at[2021, "included_provider_body_field"]) == (1, 1)
    assert years.loc[[2020], "query_returned_stories"].item() == 0 and years.query_returned_stories.sum() == 1


def test_coverage_and_corrected_invariants_fail_closed(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[Any, ...]] = []
    coverage = cohort._coverage

    def record(*args: Any) -> pd.DataFrame:
        captured.append(args)
        return coverage(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cohort, "_coverage", record)
        _run(world, "invariant-capture")
    units, summaries, bridge, members, bridged = captured[0]
    missing = dict(list(summaries.items())[1:])
    with pytest.raises(DataReadinessError, match="exactly one verified summary"):
        coverage(units, missing, bridge, members, bridged)
    outside = [{**units[0], "ledger": {**units[0]["ledger"], "requested_end_utc": pd.Timestamp("2019-01-01T00:00:00Z")}},
               *units[1:]]
    with pytest.raises(DataReadinessError, match="cover part of the initial-fit window"):
        coverage(outside, summaries, bridge, members, bridged)
    monkeypatch.setattr(cohort, "map_news_coverage", lambda frame, table: map_news_coverage(frame, table).iloc[1:])
    with pytest.raises(DataReadinessError, match="lost a ledger unit"):
        coverage(units, summaries, bridge, members, bridged)
    monkeypatch.setattr(cohort, "map_news_coverage", map_news_coverage)
    spec = json.loads((world["root"] / "configs/monthly.json").read_text())["sources"]["corrected"]
    source = _source(world["root"], spec)
    observed = next(str(row.chunk_id) for row in source["ledger"].itertuples() if row.status == "observed")
    source["records"]["collection"].pop(observed)
    with pytest.raises(DataReadinessError, match="lacks its artifact pin"):
        cohort._corrected_units(world["root"], spec, source, {})
