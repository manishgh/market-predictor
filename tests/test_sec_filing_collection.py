from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.catalysts.sec_filings import collection as sec_collection
from market_predictor.catalysts.sec_filings.collection import (
    SecFilingCollectionConfig,
    collect_historical_sec_filings,
    conservative_sec_daily_swing_availability,
    load_sec_filing_collection,
    load_sec_filing_collection_config,
    normalize_sec_identity_relations,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.sec import (
    SecFilingHistory,
    SecFilingRecord,
    SecRawResponse,
    SecSourceResponseError,
)


def _raw_response(cik: str, retrieved: datetime, *, malformed: bool = False) -> SecRawResponse:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    body = b"not-json" if malformed else json.dumps({"cik": cik}).encode()
    digest = sha256(body).hexdigest()
    response_id = sha256(
        json.dumps(
            {
                "requested_url": url,
                "final_url": url,
                "retrieved_at_utc": retrieved.isoformat(),
                "sha256": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return SecRawResponse(
        response_id=response_id,
        requested_url=url,
        final_url=url,
        status_code=200,
        retrieved_at_utc=retrieved,
        content_type="application/json",
        content_encoding="identity",
        etag=None,
        last_modified=None,
        body=body,
        body_sha256=digest,
        body_length=len(body),
        safe_headers=(("content-type", "application/json"),),
    )


class _FixtureSecSource:
    def __init__(self, *, fail_cik: str | None = None, retrieved_hour: int = 10) -> None:
        self.calls: list[str] = []
        self.fail_cik = fail_cik
        self.retrieved_hour = retrieved_hour

    def fetch_cik_filing_history(
        self,
        cik: str,
        start: datetime,
        end: datetime,
        *,
        forms: set[str] | None = None,
        ticker_hint: str = "SEC",
    ) -> SecFilingHistory:
        del start, end, forms
        self.calls.append(cik)
        observed = datetime(2026, 8, 2, self.retrieved_hour, len(self.calls), tzinfo=UTC)
        raw = _raw_response(cik, observed, malformed=cik == self.fail_cik)
        if cik == self.fail_cik:
            raise SecSourceResponseError("malformed fixture response", (raw,))
        filing = SecFilingRecord(
            ticker=ticker_hint,
            cik=cik,
            company_name="APPLE INC",
            form="8-K",
            accepted_at_utc=datetime(2026, 7, 7, 20, 5, tzinfo=UTC),
            filing_date="2026-07-07",
            report_date="2026-07-07",
            accession_number=f"{cik}-26-000001",
            primary_document="issuer-8k.htm",
            document_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/issuer-8k.htm",
            submission_file=f"CIK{cik}.json",
            file_number="001-00001",
            is_amendment=False,
            amends_accession_number=None,
            raw_sha256="a" * 64,
        )
        return SecFilingHistory(
            ticker=ticker_hint,
            cik=cik,
            company_name="APPLE INC",
            filings=(filing,),
            submission_files=(f"CIK{cik}.json",),
            response_sha256="b" * 64,
            raw_responses=(raw,),
            source_row_count=1,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _relations(*, include_failure: bool = False) -> pd.DataFrame:
    rows = [
        {
            "security_id": "security:aapl",
            "ticker": "AAPL",
            "sec_cik": "0000320193",
            "effective_from_utc": "2010-01-01T00:00:00Z",
            "effective_to_utc": None,
            "available_at_utc": "2010-01-01T00:00:00Z",
        },
        {
            "security_id": "security:aapl-class-b",
            "ticker": "AAPL.B",
            "sec_cik": "0000320193",
            "effective_from_utc": "2010-01-01T00:00:00Z",
            "effective_to_utc": None,
            "available_at_utc": "2010-01-01T00:00:00Z",
        },
    ]
    if include_failure:
        rows.append(
            {
                "security_id": "security:msft",
                "ticker": "MSFT",
                "sec_cik": "0000789019",
                "effective_from_utc": "2010-01-01T00:00:00Z",
                "effective_to_utc": None,
                "available_at_utc": "2010-01-01T00:00:00Z",
            }
        )
    return pd.DataFrame(rows)


def _config() -> SecFilingCollectionConfig:
    return SecFilingCollectionConfig(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 8),
        forms=("8-K",),
        max_workers=1,
    )


def test_repository_sec_policy_is_rate_limited_and_conservative() -> None:
    config = load_sec_filing_collection_config(Path("configs/edge_rebuild_sec_filings.toml"))
    assert 0 < config.requests_per_second < 10
    assert config.forbidden_cooldown_seconds >= 600
    assert config.rate_limit_cooldown_seconds >= 60
    assert config.dissemination_lag_minutes >= 5
    assert {"3", "3/A", "4", "4/A", "5", "5/A"}.issubset(config.forms)


def test_collection_fetches_once_per_unique_cik_and_archives_failed_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _FixtureSecSource(fail_cik="0000789019")
    original_event_frame = sec_collection._event_frame

    def assert_raw_bodies_released(results: object, *, lag_minutes: int) -> pd.DataFrame:
        issuer_results = list(results)  # type: ignore[arg-type]
        assert all(not result.raw_responses for result in issuer_results)
        assert all(result.history is None or not result.history.raw_responses for result in issuer_results)
        return original_event_frame(issuer_results, lag_minutes=lag_minutes)

    monkeypatch.setattr(sec_collection, "_event_frame", assert_raw_bodies_released)
    collection = collect_historical_sec_filings(
        _relations(include_failure=True),
        tmp_path / "sec-collection",
        source_factory=lambda: source,
        config=_config(),
        clock=_Clock(),
    )

    assert source.calls == ["0000320193", "0000789019"]
    assert collection.manifest["issuer_count"] == 2
    assert collection.manifest["failed_issuers"] == 1
    assert collection.events["sec_cik"].tolist() == ["0000320193"]
    event = collection.events.iloc[0]
    assert pd.Timestamp(event["available_at_utc"]) == pd.Timestamp("2026-07-07T20:10:00Z")
    assert pd.Timestamp(event["first_seen_at_utc"]) == pd.Timestamp("2026-08-02T10:01:00Z")
    coverage = collection.source_collections.set_index("sec_cik")
    assert coverage.loc["0000320193", "status"] == "observed"
    assert coverage.loc["0000789019", "status"] == "failed"
    assert not bool(coverage["production_eligible"].any())
    assert len(collection.raw_inventory) == 2
    assert collection.raw_inventory["issuer_error_type"].notna().sum() == 1


def test_identity_relation_drops_unproven_historical_security() -> None:
    frame = pd.DataFrame(
        [
            {
                "security_id": "cik:320193",
                "ticker": "AAPL",
                "effective_from_utc": "2020-01-01T00:00:00Z",
                "effective_to_utc": None,
                "available_at_utc": "2020-01-01T00:00:00Z",
            },
            {
                "security_id": "sp500-historical:unknown",
                "ticker": "OLD",
                "effective_from_utc": "2020-01-01T00:00:00Z",
                "effective_to_utc": None,
                "available_at_utc": "2020-01-01T00:00:00Z",
            },
        ]
    )
    relations = normalize_sec_identity_relations(frame)
    assert relations[["ticker", "sec_cik"]].to_dict(orient="records") == [{"ticker": "AAPL", "sec_cik": "0000320193"}]


def test_daily_swing_availability_moves_late_and_non_session_filings_to_next_open() -> None:
    late, late_rule = conservative_sec_daily_swing_availability(datetime(2026, 7, 7, 22, 0, tzinfo=UTC), "8-K")
    weekend, weekend_rule = conservative_sec_daily_swing_availability(datetime(2026, 7, 11, 14, 0, tzinfo=UTC), "8-K")
    assert pd.Timestamp(late) == pd.Timestamp("2026-07-08T13:30:00Z")
    assert pd.Timestamp(weekend) == pd.Timestamp("2026-07-13T13:30:00Z")
    assert late_rule == weekend_rule == "late_submission_next_xnys_open"


def test_collection_loader_rejects_raw_archive_tampering(tmp_path: Path) -> None:
    output = tmp_path / "sec-collection"
    collect_historical_sec_filings(
        _relations(),
        output,
        source_factory=lambda: _FixtureSecSource(),
        config=_config(),
        clock=_Clock(),
    )
    archive = output / "raw_responses.zip"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(DataReadinessError, match="raw response archive"):
        load_sec_filing_collection(output)


def test_collection_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "sec-collection"
    collect_historical_sec_filings(
        _relations(),
        output,
        source_factory=lambda: _FixtureSecSource(),
        config=_config(),
        clock=_Clock(),
    )
    with pytest.raises(DataReadinessError, match="immutable"):
        collect_historical_sec_filings(
            _relations(),
            output,
            source_factory=lambda: _FixtureSecSource(),
            config=_config(),
            clock=_Clock(),
        )
