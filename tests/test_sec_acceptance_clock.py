"""Per-issuer SEC acceptance clocks from real EDGAR detail pages and an end-to-end page collection; no network."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.catalysts.sec_filings import acceptance_clock as clock
from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import sec_acceptance_clock as publication
from tests.support.sec_archive import (
    PageFake,
    collect_clock_pages,
    filing,
    header_page,
    submissions,
    true_pages,
    write_clock,
    write_collection,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sec" / "clock"
NY, UTC_ = clock.NEW_YORK_LABELED_UTC, clock.UTC_LABEL
# Real pages: HollyFrontier labels New York wall-clock time as UTC in every form group; JPMorgan's labels are UTC,
# including a prospectus EDGAR accepted at 02:37 on a Saturday.
REAL = {
    "0000048039-19-000043": ("2019-08-01T06:13:52.000Z", NY),
    "0000048039-20-000015": ("2020-02-20T17:31:29.000Z", NY),
    "0001209191-19-044813": ("2019-08-06T16:22:29.000Z", NY),
    "0000891092-19-010013": ("2019-09-28T06:37:53.000Z", UTC_),
    "0001225208-19-010367": ("2019-07-18T21:11:49.000Z", UTC_),
    "0000019617-19-000127": ("2019-08-05T22:19:56.000Z", UTC_),
}
A, B, C = "0000000001", "0000000002", "0000000003"


@pytest.mark.parametrize("accession", sorted(REAL))
def test_real_pages_decide_each_issuer_convention(accession: str) -> None:
    raw, expected = REAL[accession]
    header = documents.parse_filing_header((FIXTURES / f"{accession}-index.htm").read_bytes(), accession=accession)
    assert clock.page_convention(raw, header.accepted_at_utc) == expected
    assert clock.corrected_acceptance(raw, expected) == header.accepted_at_utc
    shifted = f"{raw[:17]}{int(raw[17:19]) ^ 1:02d}.000Z"
    assert clock.page_convention(shifted, header.accepted_at_utc) is None


def test_raw_values_parse_strictly_and_new_york_readings_respect_daylight_saving() -> None:
    assert clock.corrected_acceptance("2019-07-01T16:05:00.000Z", NY) == pd.Timestamp("2019-07-01T20:05:00Z")
    assert clock.corrected_acceptance("2019-12-02T16:05:00.000Z", NY) == pd.Timestamp("2019-12-02T21:05:00Z")
    assert clock.corrected_acceptance("2019-11-03T01:30:00.000Z", UTC_) == pd.Timestamp("2019-11-03T01:30:00Z")
    for raw in ("2019-11-03T01:30:00.000Z", "2019-03-10T02:30:00.000Z"):
        with pytest.raises(DataReadinessError, match="not a New York instant"):
            clock.corrected_acceptance(raw, NY)
    for raw in ("2019-07-01T16:05:00Z", "2019-07-01T16:05:00.120Z", "20190701160500", "2019-07-01 16:05:00.000Z"):
        with pytest.raises(DataReadinessError, match="unexpected form"):
            clock.wall_clock(raw)
    with pytest.raises(DataReadinessError, match="not decided"):
        clock.corrected_acceptance("2019-07-01T16:05:00.000Z", clock.UNKNOWN)
    assert [clock.form_group(form) for form in ("8-K/A", "6-K", "10-K", "20-F", "4/A", "3", "424B2", "DEF 14A")] == [
        "current", "current", "periodic", "periodic", "ownership", "ownership", "other", "other"]


def _filings(*rows: tuple[str, str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sec_cik", "accession_number", "sec_form", "filing_date", "acceptance_raw"])


def test_sample_takes_first_and_last_filing_per_issuer_and_group() -> None:
    filings = _filings((A, "a3", "8-K", "2019-08-03", "2019-08-03T14:00:00.000Z"),
                       (A, "a1", "8-K", "2019-08-01", "2019-08-01T14:00:00.000Z"),
                       (A, "a2", "8-K/A", "2019-08-02", "2019-08-02T14:00:00.000Z"),
                       (A, "a4", "4", "2019-08-04", "2019-08-04T14:00:00.000Z"),
                       (B, "a1", "8-K", "2019-08-01", "2019-08-01T14:00:00.000Z"))
    sample = clock.clock_sample(filings)
    assert list(zip(sample.sec_cik, sample.form_group, sample.accession_number, strict=True)) == [
        (A, "current", "a1"), (A, "current", "a3"), (A, "ownership", "a4"), (B, "current", "a1")]


def test_in_data_counts_use_only_early_filings_and_treat_impossible_readings_as_outside() -> None:
    filings = _filings((A, "a1", "8-K", "2019-08-01", "2019-08-01T03:30:00.000Z"),  # 23:30 New York as UTC.
                       (A, "a2", "8-K", "2019-11-03", "2019-11-03T01:30:00.000Z"),  # Ambiguous as New York.
                       (A, "a3", "4", "2019-08-05", "2019-08-05T14:00:00.000Z"),
                       (A, "a4", "8-K", "2024-05-28", "2024-05-28T03:00:00.000Z"))  # Dated on the cutoff day.
    counts = clock.in_data_counts(filings, date(2024, 5, 28)).set_index("form_group")
    assert counts.loc["current", ["filings", "outside_as_utc", "outside_as_new_york"]].tolist() == [2, 2, 2]
    assert counts.loc["ownership", ["filings", "outside_as_utc", "outside_as_new_york"]].tolist() == [1, 0, 0]


def _pages(*rows: tuple[str, str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["form_group", "page_state", "page_convention"])


def _counts(*rows: tuple[str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["form_group", "outside_as_utc", "outside_as_new_york"])


def test_decision_needs_agreeing_pages_in_every_group_and_uncontradicted_counts() -> None:
    both = ("current", "ownership")
    assert clock.decide(A, both, _pages(("current", "archived", NY), ("ownership", "archived", NY)),
                        _counts(("current", 3, 0), ("ownership", 0, 0)))[0] == NY
    assert clock.decide(A, both, _pages(("current", "archived", UTC_), ("ownership", "missing", None)), _counts()) == (
        clock.UNKNOWN, "no readable EDGAR page for form groups ownership")
    with pytest.raises(DataReadinessError, match="disagree"):
        clock.decide(A, both, _pages(("current", "archived", NY), ("ownership", "archived", UTC_)), _counts())
    with pytest.raises(DataReadinessError, match="neither reading"):
        clock.decide(A, both, _pages(("current", "archived", None), ("ownership", "archived", UTC_)), _counts())
    with pytest.raises(DataReadinessError, match="contradict"):
        clock.decide(A, both, _pages(("current", "archived", UTC_), ("ownership", "archived", UTC_)),
                     _counts(("ownership", 1, 0)))


def _archive() -> dict[str, dict[str, Any]]:
    pages = submissions(A, "ALPHA CORP", [
        filing("0000000001-19-000001", "8-K", "2019-08-01T20:05:00Z", items="2.02"),
        filing("0000000001-19-000002", "8-K", "2019-09-05T23:30:00Z", items="8.01"),  # 19:30 New York.
        filing("0000000001-19-000003", "4", "2019-08-02T21:30:00Z")])
    pages |= submissions(B, "BETA CORP", [  # Labels are New York wall-clock time.
        filing("0000000002-19-000001", "8-K", "2019-08-01T07:30:00Z", items="2.02"),
        filing("0000000002-20-000002", "8-K", "2020-02-03T16:05:00Z", items="2.02"),
        filing("0000000002-19-000003", "4", "2019-08-05T18:00:00Z")])
    pages |= submissions(C, "GAMMA CORP", [filing("0000000003-19-000001", "8-K", "2019-08-01T14:00:00Z", items="8.01")])
    return pages


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("sec-clock") / "repo"
    relations = pd.DataFrame([{"security_id": f"cik:{cik}", "ticker": ticker, "sec_cik": cik,
                               "effective_from_utc": pd.Timestamp("2019-07-09T04:00:00Z"), "effective_to_utc": pd.NaT,
                               "available_at_utc": pd.Timestamp("2019-07-09T04:00:00Z")}
                              for cik, ticker in ((A, "AAA"), (B, "BBB"), (C, "CCC"))])
    relations["effective_to_utc"] = pd.to_datetime(relations.effective_to_utc, utc=True)
    collection = write_collection(root / "data/external/sec", _archive(), relations, ("8-K", "4"))
    return {"root": root, "collection": collection.directory}


@pytest.fixture(autouse=True)
def _runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)


def _truth(**changes: tuple[str, str] | int) -> PageFake:
    return PageFake({**true_pages(_archive(), frozenset({B}), {"0000000003-19-000001": 404}), **changes})


def test_publication_decides_each_issuer_from_its_pages(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _truth()
    pin = write_clock(world["root"], world["collection"], fake, monkeypatch, "clock")
    conventions, manifest = publication.load_sec_acceptance_clock(world["root"], pin, {})
    assert conventions == {A: UTC_, B: NY, C: clock.UNKNOWN}
    assert manifest["totals"]["unknown_issuers"] == [{"sec_cik": C, "reason": "no readable EDGAR page for form groups current"}]
    assert manifest["totals"]["pages_by_state"] == {"archived": 6, "missing": 1}
    assert len(fake.urls) == 7 and all(url.endswith("-index.htm") for url in fake.urls)
    output = world["root"] / "data/research/clock"
    counts = pd.read_parquet(output / "in_data_counts.parquet").set_index(["sec_cik", "form_group"])
    assert counts.loc[(B, "current"), ["outside_as_utc", "outside_as_new_york"]].tolist() == [1, 0]
    assert counts.loc[(A, "current"), ["outside_as_utc", "outside_as_new_york"]].tolist() == [0, 1]
    pages = pd.read_parquet(output / "pages.parquet")
    assert set(pages.loc[pages.sec_cik.eq(B), "page_convention"]) == {NY}
    with pytest.raises(FileExistsError):
        publication.publish_sec_acceptance_clock(root=world["root"], collection=manifest["sec_collection"],
                                                 pages=manifest["pages"], output=output)


@pytest.mark.parametrize(("name", "changes", "message"), [
    ("disagree", {"0000000002-19-000003": ("2019-08-05", "2019-08-05 14:00:00")}, "disagree"),
    ("neither", {"0000000001-19-000001": ("2019-08-01", "2019-08-01 16:06:00")}, "neither reading"),
    ("contrary", {"0000000001-19-000001": ("2019-08-01", "2019-08-01 20:05:00"),
                  "0000000001-19-000002": ("2019-09-05", "2019-09-05 23:30:00"),
                  "0000000001-19-000003": ("2019-08-02", "2019-08-02 21:30:00")}, "contradict"),
])
def test_publication_fails_when_the_per_issuer_model_breaks(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                            name: str, changes: dict[str, tuple[str, str]], message: str) -> None:
    with pytest.raises(DataReadinessError, match=message):
        write_clock(world["root"], world["collection"], _truth(**changes), monkeypatch, f"broken-{name}")
    assert not (world["root"] / f"data/research/broken-{name}").exists()


def test_pages_must_belong_to_the_same_archive_and_sample(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    pin = write_clock(world["root"], world["collection"], _truth(), monkeypatch, "sample")
    manifest = json.loads((world["root"] / pin["path"]).read_text())
    request = world["root"] / "data/raw/sample_pages/_request.json"
    original = request.read_bytes()
    try:
        changed = json.loads(original)
        changed["work_list_sha256"] = "0" * 64
        request.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(DataReadinessError):
            publication.publish_sec_acceptance_clock(root=world["root"], collection=manifest["sec_collection"],
                                                     pages=manifest["pages"], output=world["root"] / "data/research/other")
    finally:
        request.write_bytes(original)


def test_a_403_stops_every_worker_and_resumes_after_the_cooldown(world: dict[str, Any],
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _truth(**{"0000000002-19-000001": 403})
    stopped = collect_clock_pages(world["root"], world["collection"], fake, monkeypatch, "stopped")
    assert (stopped["status"], stopped["stop_status_code"]) == ("stopped", 403)
    output = world["root"] / "data/raw/stopped_pages"
    receipts = pd.concat([pd.read_parquet(path) for path in sorted((output / "shards").glob("*.parquet"))])
    assert receipts.state.eq("stopped").sum() == 1 and len(receipts) <= len(fake.pages)
    checkpoint = json.loads((output / "_checkpoint.json").read_text())
    assert datetime.fromisoformat(checkpoint["cooldown_until_utc"]) > datetime.now(UTC) + timedelta(minutes=9)
    with pytest.raises(DataReadinessError, match="cooldown has not elapsed"):
        collect_clock_pages(world["root"], world["collection"], fake, monkeypatch, "stopped", stopped["checkpoint_sha256"])
    fake.pages.update(_truth().pages)
    original = documents.collect
    monkeypatch.setattr(documents, "collect",
                        lambda **kwargs: original(**kwargs, now=lambda: datetime.now(UTC) + timedelta(hours=1)))
    resumed = collect_clock_pages(world["root"], world["collection"], fake, monkeypatch, "stopped",
                                  stopped["checkpoint_sha256"])
    assert resumed["status"] == "complete" and resumed["totals"]["attempts_by_state"]["stopped"] == 1
    assert resumed["totals"]["units_by_phase_and_state"] == {"header/archived": 6, "header/missing": 1}


@pytest.mark.parametrize(("name", "page", "outcome"), [
    ("maintenance", b"<html>EDGAR is temporarily unavailable</html>", "unknown"),
    ("foreign", None, "fail"),
])
def test_an_unreadable_page_is_unknown_but_a_foreign_page_fails(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                                 name: str, page: bytes | None, outcome: str) -> None:
    accession = "0000000002-19-000003"  # BETA's only ownership filing.
    body = page if page is not None else header_page("0000000009-19-000009", "2019-08-05", "2019-08-05 18:00:00")
    served = PageFake({**_truth().pages, accession: body})
    if outcome == "fail":
        with pytest.raises(DataReadinessError, match="other accessions"):
            write_clock(world["root"], world["collection"], served, monkeypatch, f"page-{name}")
        return
    pin = write_clock(world["root"], world["collection"], served, monkeypatch, f"page-{name}")
    conventions, manifest = publication.load_sec_acceptance_clock(world["root"], pin, {})
    assert conventions[B] == clock.UNKNOWN and conventions[A] == UTC_
    pages = pd.read_parquet(world["root"] / f"data/research/page-{name}/pages.parquet").set_index("accession_number")
    assert (pages.loc[accession, "page_state"], pages.loc[accession, "page_reason"]) == (
        "rejected", "filing index lacks its accession number")
    assert {"sec_cik": B, "reason": "no readable EDGAR page for form groups ownership"} in manifest["totals"]["unknown_issuers"]


def test_a_joint_filing_is_compared_with_each_issuer_raw_value_and_later_pages_are_counted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One accession filed by an issuer labelling UTC and one labelling New York wall-clock time; a third issuer
    has only a filing after the initial-fit cutoff, decided by its page alone."""
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)
    root = tmp_path / "repo"
    joint = "0000000009-19-000001"
    pages = submissions(A, "ALPHA CORP", [filing(joint, "8-K", "2019-08-01T20:05:00Z", items="2.02")])
    pages |= submissions(B, "BETA CORP", [filing(joint, "8-K", "2019-08-01T16:05:00Z", items="2.02")])
    pages |= submissions(C, "GAMMA CORP", [filing("0000000003-25-000001", "8-K", "2025-03-03T21:05:00Z", items="8.01")])
    relations = pd.DataFrame([{"security_id": f"cik:{cik}", "ticker": ticker, "sec_cik": cik,
                               "effective_from_utc": pd.Timestamp("2019-07-09T04:00:00Z"), "effective_to_utc": pd.NaT,
                               "available_at_utc": pd.Timestamp("2019-07-09T04:00:00Z")}
                              for cik, ticker in ((A, "AAA"), (B, "BBB"), (C, "CCC"))])
    relations["effective_to_utc"] = pd.to_datetime(relations.effective_to_utc, utc=True)
    collection = write_collection(root / "data/external/sec", pages, relations, ("8-K",))
    fake = PageFake({joint: ("2019-08-01", "2019-08-01 16:05:00"), "0000000003-25-000001": ("2025-03-03", "2025-03-03 16:05:00")})
    pin = write_clock(root, collection.directory, fake, monkeypatch, "joint")
    conventions, _ = publication.load_sec_acceptance_clock(root, pin, {})
    assert conventions == {A: UTC_, B: NY, C: UTC_}
    assert len(fake.urls) == 2
    issuers = pd.read_parquet(root / "data/research/joint/issuers.parquet").set_index("sec_cik")
    assert issuers.pages_after_initial_fit_cutoff.to_dict() == {A: 0, B: 0, C: 1}
    counts = pd.read_parquet(root / "data/research/joint/in_data_counts.parquet")
    assert C not in set(counts.sec_cik)
