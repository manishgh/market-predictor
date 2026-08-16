from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import pytest

from market_predictor.edge_rebuild import sp500_observed_memberships as observed
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.spglobal_archive import SEARCH_PAGE_SIZE, SpGlobalAnnouncement
from market_predictor.v3.universe import IndexChange, parse_sp500_changes


def _base_memberships() -> pd.DataFrame:
    effective_from = pd.Timestamp("2020-01-02T05:00:00Z")
    return pd.DataFrame(
        [
            {
                "ticker": "OLD",
                "security_id": "cik:0000000001",
                "effective_from_utc": effective_from,
                "effective_to_utc": pd.NaT,
                "available_at_utc": effective_from,
                "sector": "Industrials",
                "industry": "Machinery",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": "XLI",
                "universe_snapshot_id": "closed",
            },
            {
                "ticker": "KEEP",
                "security_id": "cik:0000000002",
                "effective_from_utc": effective_from,
                "effective_to_utc": pd.NaT,
                "available_at_utc": effective_from,
                "sector": "Information Technology",
                "industry": "Software",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "closed",
            },
        ]
    )


def _anchor(*tickers: str) -> pd.DataFrame:
    identities = {
        "OLD": "0000000001",
        "KEEP": "0000000002",
        "NEW": "0000000003",
    }
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "company": f"{ticker} Company",
                "sector": (
                    "Information Technology" if ticker == "KEEP" else "Industrials"
                ),
                "industry": "Software" if ticker == "KEEP" else "Machinery",
                "cik": identities[ticker],
            }
            for ticker in tickers
        ]
    )


def _change(
    action: str,
    ticker: str,
    effective_at: datetime,
    *,
    source_url: str = "https://press.spglobal.com/release",
    company: str | None = None,
) -> IndexChange:
    return IndexChange(
        effective_at_utc=effective_at,
        action=action,
        ticker=ticker,
        company=company or f"{ticker} Company",
        sector="Industrials",
        source_url=source_url,
        source_published_date=date(2026, 8, 13),
        source_sha256=hashlib.sha256(source_url.encode()).hexdigest(),
    )


def _raw_unit(root: Path, name: str, body: bytes, *, final_url: str) -> dict[str, object]:
    path = root / "objects" / f"{name}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {
        "body_path": path.relative_to(root).as_posix(),
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "content_encoding": "",
        "content_type": "text/html; charset=utf-8",
        "final_url": final_url,
    }


def test_sec_identity_overrides_erroneous_membership_source_cik(tmp_path: Path) -> None:
    rows = []
    sec_records: dict[str, dict[str, object]] = {}
    for number in range(500):
        ticker = f"T{number:03d}"
        wrong_cik = "9999999999" if ticker == "T007" else f"{number + 1:010d}"
        rows.append(
            "<tr>"
            f"<td>{ticker}</td><td>{ticker} Company</td>"
            "<td>Industrials</td><td>Machinery</td>"
            f"<td>{wrong_cik}</td>"
            "</tr>"
        )
        sec_records[str(number)] = {"ticker": ticker, "cik_str": 100_000 + number}
    for number in range(4_500):
        sec_records[str(500 + number)] = {
            "ticker": f"X{number:04d}",
            "cik_str": 1_000_000 + number,
        }
    html = (
        "<html><body><table id='constituents'>"
        "<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>"
        "<th>GICS Sub-Industry</th><th>CIK</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    ).encode()
    anchor_unit = _raw_unit(
        tmp_path,
        "anchor",
        html,
        final_url=observed.ANCHOR_URL,
    )
    identity_unit = _raw_unit(
        tmp_path,
        "identities",
        json.dumps(sec_records).encode(),
        final_url=observed.SEC_IDENTITY_URL,
    )

    anchor = observed._parse_anchor(tmp_path, anchor_unit, identity_unit)

    actual = anchor.loc[anchor["ticker"].eq("T007"), "cik"].item()
    assert actual == "0000100007"
    assert actual != "9999999999"


def test_closed_future_effective_event_applies_when_observation_reaches_it() -> None:
    effective_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)

    result = observed._extend_memberships(
        _base_memberships(),
        base_cutoff=date(2026, 8, 15),
        observed_at=datetime(2026, 8, 18, 15, tzinfo=UTC),
        closed_changes=(
            _change("deletion", "OLD", effective_at),
            _change("addition", "NEW", effective_at),
        ),
        observed_changes=(),
        anchor=_anchor("KEEP", "NEW"),
    )

    old = result[result["ticker"].eq("OLD")].iloc[-1]
    new = result[result["ticker"].eq("NEW")].iloc[-1]
    assert old["effective_to_utc"] == pd.Timestamp(effective_at)
    assert new["effective_from_utc"] == pd.Timestamp(effective_at)
    assert new["available_at_utc"] == pd.Timestamp(effective_at)
    assert new["security_id"] == "cik:0000000003"


def test_future_effective_changes_remain_in_pending_inventory() -> None:
    future = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    pending = observed._pending_changes(
        (
            _change("deletion", "OLD", future),
            _change("addition", "NEW", future),
        ),
        (),
        observed_at=datetime(2026, 8, 17, 15, tzinfo=UTC),
    )

    assert [(item.action, item.ticker) for item in pending] == [
        ("addition", "NEW"),
        ("deletion", "OLD"),
    ]


def test_same_date_event_does_not_apply_before_exact_effective_time() -> None:
    effective_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    observed_at = datetime(2026, 8, 18, 12, tzinfo=UTC)
    changes = (
        _change("deletion", "OLD", effective_at),
        _change("addition", "NEW", effective_at),
    )

    result = observed._extend_memberships(
        _base_memberships(),
        base_cutoff=date(2026, 8, 15),
        observed_at=observed_at,
        closed_changes=changes,
        observed_changes=(),
        anchor=_anchor("OLD", "KEEP"),
    )

    assert set(result.loc[result["effective_to_utc"].isna(), "ticker"]) == {
        "OLD",
        "KEEP",
    }
    assert len(observed._pending_changes(changes, (), observed_at=observed_at)) == 2


def test_observed_non_membership_release_has_explicit_empty_outcome() -> None:
    changes = parse_sp500_changes(
        "<html><body><p>Quarterly index market commentary.</p></body></html>",
        source_url="https://press.spglobal.com/2026-08-17-market-commentary",
        published_date=date(2026, 8, 17),
        allow_verified_no_membership_event=True,
    )

    assert changes == []


def test_malformed_membership_table_cannot_be_classified_as_no_event() -> None:
    html = """
    <html><body><table>
      <tr><th>Index Name</th><th>Action</th><th>Ticker</th></tr>
      <tr><td>S&amp;P 500</td><td>Addition</td><td>NEW</td></tr>
    </table></body></html>
    """

    with pytest.raises(DataReadinessError, match="no structured S&P 500 change rows"):
        parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2026-08-17-malformed",
            published_date=date(2026, 8, 17),
            allow_verified_no_membership_event=True,
            source_title="S&P 500 Changes",
        )


def test_generic_change_title_and_malformed_add_row_fail_closed() -> None:
    html = """
    <html><body><table>
      <tr><th>Index</th><th>Action</th><th>Company</th></tr>
      <tr><td>S&amp;P 500</td><td>Add</td><td>New Company</td></tr>
    </table></body></html>
    """

    with pytest.raises(DataReadinessError, match="no structured S&P 500 change rows"):
        parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2026-08-17-generic-change",
            published_date=date(2026, 8, 17),
            source_sha256="b" * 64,
            allow_verified_no_membership_event=True,
            source_title="Changes to S&P 500",
        )


def test_same_day_observed_event_is_not_available_before_observation() -> None:
    effective_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    observed_at = datetime(2026, 8, 18, 15, tzinfo=UTC)

    result = observed._extend_memberships(
        _base_memberships(),
        base_cutoff=date(2026, 8, 15),
        observed_at=observed_at,
        closed_changes=(),
        observed_changes=(
            _change("deletion", "OLD", effective_at),
            _change("addition", "NEW", effective_at),
        ),
        anchor=_anchor("KEEP", "NEW"),
    )

    new = result[result["ticker"].eq("NEW")].iloc[-1]
    assert new["effective_from_utc"] == pd.Timestamp(effective_at)
    assert new["available_at_utc"] == pd.Timestamp(observed_at)


def test_quiet_day_anchor_mismatch_fails_closed() -> None:
    with pytest.raises(DataReadinessError, match="state differs from independent anchor"):
        observed._extend_memberships(
            _base_memberships(),
            base_cutoff=date(2026, 8, 15),
            observed_at=datetime(2026, 8, 17, 15, tzinfo=UTC),
            closed_changes=(),
            observed_changes=(),
            anchor=_anchor("KEEP", "NEW"),
        )


def test_unbalanced_observed_event_batch_fails_closed() -> None:
    effective_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    with pytest.raises(DataReadinessError, match="change batch is unbalanced"):
        observed._extend_memberships(
            _base_memberships(),
            base_cutoff=date(2026, 8, 15),
            observed_at=datetime(2026, 8, 18, 15, tzinfo=UTC),
            closed_changes=(),
            observed_changes=(_change("deletion", "OLD", effective_at),),
            anchor=_anchor("KEEP"),
        )


def test_closed_and_observed_versions_of_same_event_must_agree() -> None:
    effective_at = datetime(2026, 8, 18, 13, 30, tzinfo=UTC)
    with pytest.raises(DataReadinessError, match="closed and observed S&P events conflict"):
        observed._extend_memberships(
            _base_memberships(),
            base_cutoff=date(2026, 8, 15),
            observed_at=datetime(2026, 8, 18, 15, tzinfo=UTC),
            closed_changes=(
                _change(
                    "addition",
                    "NEW",
                    effective_at,
                    source_url="https://press.spglobal.com/closed",
                ),
            ),
            observed_changes=(
                _change(
                    "addition",
                    "NEW",
                    effective_at,
                    source_url="https://press.spglobal.com/observed",
                ),
            ),
            anchor=_anchor("OLD", "KEEP", "NEW"),
        )


def _dated_page(*, prefix: str, published: date) -> list[tuple[date, str]]:
    return [
        (published, f"https://press.spglobal.com/{prefix}-{number:03d}")
        for number in range(SEARCH_PAGE_SIZE)
    ]


def test_pagination_requires_exact_overlap_and_newest_first() -> None:
    first = _dated_page(prefix="first", published=date(2026, 8, 15))
    observed._validate_page_overlap(0, first, None)
    second = [
        first[-1],
        *_dated_page(prefix="second", published=date(2026, 8, 14))[:-1],
    ]
    observed._validate_page_overlap(1, second, first[-1][1])

    with pytest.raises(DataReadinessError, match="pagination overlap"):
        observed._validate_page_overlap(1, second, "https://press.spglobal.com/wrong")

    out_of_order = first.copy()
    out_of_order[1] = (date(2026, 8, 16), out_of_order[1][1])
    with pytest.raises(DataReadinessError, match="newest-to-oldest"):
        observed._validate_page_overlap(0, out_of_order, None)


@dataclass(frozen=True)
class _Response:
    body: bytes
    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    status_code: int
    retrieved_at_utc: datetime
    content_type: str | None
    content_encoding: str | None
    etag: str | None
    last_modified: str | None
    body_length: int
    sha256: str
    body_representation: str


def _response(
    requested_url: str,
    *,
    final_url: str | None = None,
    redirect_chain: tuple[str, ...] = (),
) -> _Response:
    body = b"response"
    return _Response(
        body=body,
        requested_url=requested_url,
        final_url=final_url or requested_url,
        redirect_chain=redirect_chain,
        status_code=200,
        retrieved_at_utc=datetime(2026, 8, 18, 15, tzinfo=UTC),
        content_type="text/html",
        content_encoding=None,
        etag=None,
        last_modified=None,
        body_length=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        body_representation="http_entity_encoded",
    )


def test_http_identity_accepts_query_reordering_but_rejects_redirects() -> None:
    requested = "https://press.spglobal.com/index.php?b=2&a=1"
    observed._verify_response(
        _response(requested),
        expected_url="https://press.spglobal.com/index.php?a=1&b=2",
        expected_host="press.spglobal.com",
        exact_path="/index.php",
    )

    with pytest.raises(DataReadinessError, match="HTTP response identity"):
        observed._verify_response(
            _response(
                requested,
                final_url="https://press.spglobal.com/redirected",
                redirect_chain=(requested,),
            ),
            expected_url="https://press.spglobal.com/index.php?a=1&b=2",
            expected_host="press.spglobal.com",
            exact_path="/index.php",
        )

    with pytest.raises(DataReadinessError, match="HTTP response identity"):
        observed._verify_response(
            _response("https://user@press.spglobal.com:444/index.php?a=1&b=2"),
            expected_url="https://press.spglobal.com/index.php?a=1&b=2",
            expected_host="press.spglobal.com",
            exact_path="/index.php",
        )


def test_race_identity_changes_when_first_page_semantics_change() -> None:
    urls = _dated_page(prefix="race", published=date(2026, 8, 15))
    first = [
        SpGlobalAnnouncement(
            published_date=date(2026, 8, 15),
            title="Original release",
            url=urls[0][1],
            origin="discovered",
        )
    ]
    confirmation = [
        SpGlobalAnnouncement(
            published_date=date(2026, 8, 15),
            title="Changed release",
            url=urls[0][1],
            origin="discovered",
        )
    ]

    assert observed._page_semantics(first, urls) != observed._page_semantics(
        confirmation,
        urls,
    )


def test_path_and_response_body_tampering_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="escapes authority root"):
        observed._resolve_inside(tmp_path, "../outside.bin")

    unit = _raw_unit(
        tmp_path,
        "body",
        b"original",
        final_url="https://press.spglobal.com/release",
    )
    assert observed._unit_body(tmp_path, unit) == b"original"
    (tmp_path / str(unit["body_path"])).write_bytes(b"modified")
    with pytest.raises(DataReadinessError, match="response body changed"):
        observed._unit_body(tmp_path, unit)


def test_artifact_hash_and_immutable_json_helpers_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    record: dict[str, Any] = {
        "path": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": artifact.stat().st_size,
    }
    assert observed._verified_artifact(tmp_path, record) == artifact
    artifact.write_text("[]", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="artifact changed"):
        observed._verified_artifact(tmp_path, record)

    immutable = tmp_path / "immutable.json"
    observed._write_new_json(immutable, {"value": 1})
    with pytest.raises(DataReadinessError, match="attempt is immutable"):
        observed._write_new_json(immutable, {"value": 2})


def test_extra_root_file_fails_raw_inventory(tmp_path: Path) -> None:
    (tmp_path / "objects").mkdir()
    (tmp_path / "units").mkdir()
    expected_files = {
        "_request.json",
        "_status.json",
        "_manifest.json",
        "_authority.json",
        "_collector.lock",
        observed.ANCHOR_FILE,
        observed.EVENT_FILE,
        observed.OUTCOME_FILE,
        observed.PENDING_FILE,
        observed.MEMBERSHIP_FILE,
        "memberships.parquet.manifest.json",
        "memberships.parquet.lock",
    }
    for name in expected_files:
        (tmp_path / name).write_bytes(b"")
    observed._verify_unit_inventory(tmp_path, [])

    (tmp_path / "unexpected.txt").write_text("poison", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="root files differ"):
        observed._verify_unit_inventory(tmp_path, [])


@dataclass(frozen=True)
class _ClosedEvents:
    authority_sha256: str
    event_set_sha256: str
    changes: tuple[IndexChange, ...]


class _ObservedMembershipHttpClient:
    def __init__(self, bodies: dict[str, bytes], observed_at: datetime) -> None:
        self.bodies = bodies
        self.observed_at = observed_at
        self.allow_redirects_values: list[bool] = []
        self.calls: list[str] = []

    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = observed.MAXIMUM_RESPONSE_BYTES,
        allow_redirects: bool = False,
    ) -> _Response:
        del retries, pause
        self.allow_redirects_values.append(allow_redirects)
        if allow_redirects:
            raise AssertionError("observed membership HTTP must disable redirects")
        requested_url = url if params is None else f"{url}?{urlencode(params)}"
        body = self.bodies[url]
        if len(body) > maximum_body_bytes:
            raise AssertionError("fake response exceeds configured body limit")
        self.calls.append(requested_url)
        return _Response(
            body=body,
            requested_url=requested_url,
            final_url=requested_url,
            redirect_chain=(),
            status_code=200,
            retrieved_at_utc=self.observed_at,
            content_type="application/json" if url == observed.SEC_IDENTITY_URL else "text/html; charset=utf-8",
            content_encoding=None,
            etag='"fixture"',
            last_modified=None,
            body_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            body_representation="http_entity_encoded",
        )


class _SecondPageRaceHttpClient(_ObservedMembershipHttpClient):
    def __init__(self, anchor_body: bytes, sec_body: bytes) -> None:
        super().__init__(
            {
                observed.ANCHOR_URL: anchor_body,
                observed.SEC_IDENTITY_URL: sec_body,
            },
            datetime(2026, 8, 17, 15, tzinfo=UTC),
        )
        self.archive_calls = {0: 0, 1: 0}
        first_urls = [
            f"https://press.spglobal.com/2026-08-17-race-{number:03d}"
            for number in range(SEARCH_PAGE_SIZE)
        ]
        self.first_page = _search_page_body(first_urls)
        second_urls = [
            first_urls[-1],
            *[
                f"https://press.spglobal.com/2026-08-15-race-{number:03d}"
                for number in range(SEARCH_PAGE_SIZE - 1)
            ],
        ]
        self.second_page = _search_page_body(second_urls)
        self.changed_second_page = _search_page_body(
            second_urls,
            changed_title_at=1,
        )

    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = observed.MAXIMUM_RESPONSE_BYTES,
        allow_redirects: bool = False,
    ) -> _Response:
        if url != observed.SP_GLOBAL_ARCHIVE_URL and url not in self.bodies:
            self.bodies[url] = b"<html><body><p>Index commentary.</p></body></html>"
        if url != observed.SP_GLOBAL_ARCHIVE_URL:
            return super().get_bytes_with_metadata(
                url,
                params=params,
                retries=retries,
                pause=pause,
                maximum_body_bytes=maximum_body_bytes,
                allow_redirects=allow_redirects,
            )
        page_number = int((params or {})["o"]) // observed.SEARCH_PAGE_STRIDE
        self.archive_calls[page_number] += 1
        body = self.first_page if page_number == 0 else self.second_page
        if page_number == 1 and self.archive_calls[page_number] == 2:
            body = self.changed_second_page
        self.bodies[url] = body
        return super().get_bytes_with_metadata(
            url,
            params=params,
            retries=retries,
            pause=pause,
            maximum_body_bytes=maximum_body_bytes,
            allow_redirects=allow_redirects,
        )


def _search_page_body(
    urls: list[str],
    *,
    changed_title_at: int | None = None,
) -> bytes:
    links = "".join(
        f'<a href="{url}">'
        f'{"Changed archive item" if number == changed_title_at else "Unrelated archive item"}'
        "</a>"
        for number, url in enumerate(urls)
    )
    return f"<html><body>{links}</body></html>".encode()


def _large_observed_membership_fixture() -> tuple[pd.DataFrame, bytes, bytes]:
    effective_from = pd.Timestamp("2020-01-02T05:00:00Z")
    memberships: list[dict[str, object]] = []
    anchor_rows: list[str] = []
    sec_records: dict[str, dict[str, object]] = {}
    for number in range(500):
        ticker = f"T{number:03d}"
        cik = f"{100_000 + number:010d}"
        memberships.append(
            {
                "ticker": ticker,
                "security_id": f"cik:{cik}",
                "effective_from_utc": effective_from,
                "effective_to_utc": pd.NaT,
                "available_at_utc": effective_from,
                "sector": "Industrials",
                "industry": "Machinery",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": "XLI",
                "universe_snapshot_id": "closed",
                "source": "spglobal_official_point_in_time",
                "availability_policy": "provider_publication_proxy",
                "schema_version": "market_data.v1",
            }
        )
        anchor_rows.append(
            "<tr>"
            f"<td>{ticker}</td><td>{ticker} Company</td>"
            "<td>Industrials</td><td>Machinery</td>"
            f"<td>{cik}</td>"
            "</tr>"
        )
        sec_records[str(number)] = {"ticker": ticker, "cik_str": 100_000 + number}
    for number in range(4_500):
        sec_records[str(500 + number)] = {
            "ticker": f"X{number:04d}",
            "cik_str": 1_000_000 + number,
        }
    anchor_body = (
        "<html><body><table id='constituents'>"
        "<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>"
        "<th>GICS Sub-Industry</th><th>CIK</th></tr>"
        + "".join(anchor_rows)
        + "</table></body></html>"
    ).encode()
    return pd.DataFrame(memberships), anchor_body, json.dumps(sec_records).encode()


def _quiet_official_search_page() -> bytes:
    links = "".join(
        (
            '<a href="https://press.spglobal.com/'
            f'2026-08-15-quiet-fixture-{number:03d}">Unrelated archive item</a>'
        )
        for number in range(SEARCH_PAGE_SIZE)
    )
    return f"<html><body>{links}</body></html>".encode()


def test_public_collect_load_round_trip_and_inventory_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, anchor_body, sec_body = _large_observed_membership_fixture()
    base_root = tmp_path / "base"
    archive_root = tmp_path / "archive"
    event_root = tmp_path / "events"
    output_root = tmp_path / "observed"
    for root in (base_root, archive_root, event_root):
        root.mkdir()
    closed_events = _ClosedEvents(
        authority_sha256="a" * 64,
        event_set_sha256="e" * 64,
        changes=(),
    )
    (base_root / "_request.json").write_text(
        json.dumps(
            {
                "parent_lineage": {
                    "event_authority_sha256": closed_events.authority_sha256,
                    "event_set_sha256": closed_events.event_set_sha256,
                }
            }
        ),
        encoding="utf-8",
    )
    base_parent: dict[str, object] = {
        "authority_sha256": "b" * 64,
        "manifest_sha256": "m" * 64,
        "membership_table_sha256": "t" * 64,
        "universe_sha256": "u" * 64,
        "cutoff_date": "2026-08-15",
    }
    monkeypatch.setattr(
        observed,
        "load_sp500_membership_authority_envelope",
        lambda _: (base.copy(), dict(base_parent)),
    )
    monkeypatch.setattr(
        observed,
        "require_spglobal_event_reconstruction_ready",
        lambda *_args, **_kwargs: closed_events,
    )
    client = _ObservedMembershipHttpClient(
        {
            observed.SP_GLOBAL_ARCHIVE_URL: _quiet_official_search_page(),
            observed.ANCHOR_URL: anchor_body,
            observed.SEC_IDENTITY_URL: sec_body,
        },
        datetime(2026, 8, 17, 15, tzinfo=UTC),
    )

    manifest = observed.collect_observed_sp500_membership_authority(
        base_membership_directory=base_root,
        closed_archive_directory=archive_root,
        closed_event_directory=event_root,
        output_directory=output_root,
        client_factory=lambda: client,
        config=observed.ObservedMembershipConfig(maximum_pages=1),
    )
    authority = observed.load_observed_sp500_membership_authority(output_root)

    assert manifest["status"] == "complete"
    assert manifest["anchor_constituent_count"] == 500
    assert manifest["sec_identity_count"] == 5_000
    assert manifest["new_release_count"] == 0
    assert manifest["effective_horizon_date"] == "2026-08-17"
    assert len(authority.memberships) == 500
    assert authority.parent["authority_type"] == "observed_time"
    assert client.allow_redirects_values == [False, False, False, False]

    status_path = output_root / "_status.json"
    original_status = status_path.read_bytes()
    poisoned_status = json.loads(original_status)
    poisoned_status["status"] = "poisoned"
    status_path.write_text(json.dumps(poisoned_status), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="envelope is invalid"):
        observed.load_observed_sp500_membership_authority(output_root)
    status_path.write_bytes(original_status)

    nested_poison = output_root / "objects" / "unexpected" / "poison.bin"
    nested_poison.parent.mkdir()
    nested_poison.write_bytes(b"poison")
    with pytest.raises(DataReadinessError, match="raw files differ from inventory"):
        observed.load_observed_sp500_membership_authority(output_root)


def test_public_collection_rejects_change_on_second_confirmation_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, anchor_body, sec_body = _large_observed_membership_fixture()
    base_root = tmp_path / "base"
    archive_root = tmp_path / "archive"
    event_root = tmp_path / "events"
    output_root = tmp_path / "observed"
    for root in (base_root, archive_root, event_root):
        root.mkdir()
    closed_events = _ClosedEvents(
        authority_sha256="a" * 64,
        event_set_sha256="e" * 64,
        changes=(),
    )
    (base_root / "_request.json").write_text(
        json.dumps(
            {
                "parent_lineage": {
                    "event_authority_sha256": closed_events.authority_sha256,
                    "event_set_sha256": closed_events.event_set_sha256,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        observed,
        "load_sp500_membership_authority_envelope",
        lambda _: (
            base.copy(),
            {
                "authority_sha256": "b" * 64,
                "manifest_sha256": "m" * 64,
                "membership_table_sha256": "t" * 64,
                "universe_sha256": "u" * 64,
                "cutoff_date": "2026-08-15",
            },
        ),
    )
    monkeypatch.setattr(
        observed,
        "require_spglobal_event_reconstruction_ready",
        lambda *_args, **_kwargs: closed_events,
    )
    client = _SecondPageRaceHttpClient(anchor_body, sec_body)

    with pytest.raises(
        DataReadinessError,
        match="official release index changed during anchor observation",
    ):
        observed.collect_observed_sp500_membership_authority(
            base_membership_directory=base_root,
            closed_archive_directory=archive_root,
            closed_event_directory=event_root,
            output_directory=output_root,
            client_factory=lambda: client,
            config=observed.ObservedMembershipConfig(maximum_pages=2),
        )

    assert client.archive_calls == {0: 2, 1: 2}
    assert not (output_root / "_authority.json").exists()
