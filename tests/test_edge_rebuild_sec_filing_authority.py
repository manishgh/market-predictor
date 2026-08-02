from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.edge_rebuild.sec_filing_authority import (
    _merge_events,
    attach_sec_filing_features,
    load_sec_filing_decision_authority,
    publish_sec_filing_decision_authority,
)
from market_predictor.edge_rebuild.sec_filing_collection import (
    SecFilingCollectionConfig,
    collect_historical_sec_filings,
    load_sec_filing_collection,
)
from market_predictor.sources.sec import SecFilingHistory, SecFilingRecord, SecRawResponse
from market_predictor.v3.errors import DataReadinessError


def _relations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "security:aapl",
                "ticker": "AAPL",
                "sec_cik": "0000320193",
                "effective_from_utc": "2010-01-01T00:00:00Z",
                "effective_to_utc": None,
                "available_at_utc": "2010-01-01T00:00:00Z",
            },
            {
                "security_id": "security:msft",
                "ticker": "MSFT",
                "sec_cik": "0000789019",
                "effective_from_utc": "2010-01-01T00:00:00Z",
                "effective_to_utc": None,
                "available_at_utc": "2010-01-01T00:00:00Z",
            },
        ]
    )


class _AuthorityFixtureSource:
    def __init__(self, observed_at: datetime) -> None:
        self.observed_at = observed_at

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
        filename = f"CIK{cik}.json"
        url = f"https://data.sec.gov/submissions/{filename}"
        body = json.dumps({"cik": cik}).encode()
        digest = sha256(body).hexdigest()
        response_id = sha256(
            json.dumps(
                {
                    "requested_url": url,
                    "final_url": url,
                    "retrieved_at_utc": self.observed_at.isoformat(),
                    "sha256": digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        raw = SecRawResponse(
            response_id=response_id,
            requested_url=url,
            final_url=url,
            status_code=200,
            retrieved_at_utc=self.observed_at,
            content_type="application/json",
            content_encoding="identity",
            etag=None,
            last_modified=None,
            body=body,
            body_sha256=digest,
            body_length=len(body),
            safe_headers=(("content-type", "application/json"),),
        )
        filings: tuple[SecFilingRecord, ...] = ()
        if cik == "0000320193":
            filings = (
                SecFilingRecord(
                    ticker=ticker_hint,
                    cik=cik,
                    company_name="APPLE INC",
                    form="8-K",
                    accepted_at_utc=datetime(2026, 7, 7, 20, 5, tzinfo=UTC),
                    filing_date="2026-07-07",
                    report_date="2026-07-07",
                    accession_number="0000320193-26-000001",
                    primary_document="aapl-8k.htm",
                    document_url="https://www.sec.gov/aapl-8k.htm",
                    submission_file=filename,
                    file_number="001-36743",
                    is_amendment=False,
                    amends_accession_number=None,
                    raw_sha256="a" * 64,
                ),
            )
        return SecFilingHistory(
            ticker=ticker_hint,
            cik=cik,
            company_name=ticker_hint,
            filings=filings,
            submission_files=(filename,),
            response_sha256="b" * 64,
            raw_responses=(raw,),
            source_row_count=len(filings),
        )


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self.value
        self.value += timedelta(seconds=1)
        return value


def _collection(tmp_path: Path, name: str = "collection", *, observed_hour: int = 10) -> Path:
    output = tmp_path / name
    source = _AuthorityFixtureSource(datetime(2026, 8, 2, observed_hour, 0, tzinfo=UTC))
    collect_historical_sec_filings(
        _relations(),
        output,
        source_factory=lambda: source,
        config=SecFilingCollectionConfig(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 8),
            forms=("8-K",),
            max_workers=1,
        ),
        clock=_Clock(),
    )
    return output


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_id": ["aapl-after", "aapl-before", "msft-zero", "goog-unknown", "august-unknown"],
            "security_id": ["security:aapl", "security:aapl", "security:msft", "security:goog", "security:aapl"],
            "ticker": ["AAPL", "AAPL", "MSFT", "GOOG", "AAPL"],
            "decision_time_utc": pd.to_datetime(
                [
                    "2026-07-08T20:00:00Z",
                    "2026-07-07T19:00:00Z",
                    "2026-07-08T20:00:00Z",
                    "2026-07-08T20:00:00Z",
                    "2026-08-03T20:00:00Z",
                ],
                utc=True,
            ),
        }
    )


def test_authority_uses_effective_relation_and_distinguishes_zero_from_unknown(tmp_path: Path) -> None:
    authority = publish_sec_filing_decision_authority(
        _decisions(),
        [_collection(tmp_path)],
        _relations(),
        tmp_path / "authority",
    )
    rows = authority.decisions.set_index("decision_id")
    assert rows.loc["aapl-after", "sec_filing_count_1d"] == 1.0
    assert rows.loc["aapl-before", "sec_filing_count_1d"] == 0.0
    assert rows.loc["msft-zero", "sec_filing_count_3d"] == 0.0
    assert pd.isna(rows.loc["goog-unknown", "sec_filing_count_3d"])
    assert not bool(rows.loc["goog-unknown", "sec_identity_proven"])
    assert pd.isna(rows.loc["august-unknown", "sec_filing_count_3d"])
    assert [record["month_et"] for record in authority.partition_records] == ["2026-07", "2026-08"]
    assert all(Path(authority.directory / str(record["path"])).is_file() for record in authority.partition_records)

    attached = attach_sec_filing_features(_decisions(), authority, require_production_ready=False)
    assert "sec_filing_count_3d" in attached.columns


def test_effective_relation_available_after_decision_fails_unknown(tmp_path: Path) -> None:
    relations = _relations()
    relations.loc[relations["ticker"].eq("AAPL"), "available_at_utc"] = "2026-07-09T00:00:00Z"
    relations.loc[relations["ticker"].eq("AAPL"), "effective_from_utc"] = "2026-07-09T00:00:00Z"
    authority = publish_sec_filing_decision_authority(
        _decisions(),
        [_collection(tmp_path)],
        relations,
        tmp_path / "authority",
    )
    row = authority.decisions.set_index("decision_id").loc["aapl-after"]
    assert not bool(row["sec_identity_proven"])
    assert pd.isna(row["sec_filing_count_1d"])


def test_overlapping_generations_keep_earliest_first_seen_for_same_accession(tmp_path: Path) -> None:
    later = _collection(tmp_path, "later", observed_hour=11)
    earlier = _collection(tmp_path, "earlier", observed_hour=9)
    merged = _merge_events([load_sec_filing_collection(later), load_sec_filing_collection(earlier)])
    assert len(merged) == 1
    assert pd.Timestamp(merged.iloc[0]["first_seen_at_utc"]) == pd.Timestamp("2026-08-02T09:00:00Z")
    authority = publish_sec_filing_decision_authority(
        _decisions(),
        [later, earlier],
        _relations(),
        tmp_path / "overlap-authority",
    )
    row = authority.decisions.set_index("decision_id").loc["aapl-after"]
    assert row["sec_filing_count_1d"] == 1.0
    assert authority.manifest["event_rows"] == 1
    assert len(authority.coverage) == 4


def test_research_authority_rejects_production_and_source_tampering(tmp_path: Path) -> None:
    collection = _collection(tmp_path)
    authority = publish_sec_filing_decision_authority(
        _decisions(),
        [collection],
        _relations(),
        tmp_path / "authority",
    )
    with pytest.raises(DataReadinessError, match="not production ready"):
        load_sec_filing_decision_authority(authority.directory)
    with pytest.raises(DataReadinessError, match="cannot produce"):
        publish_sec_filing_decision_authority(
            _decisions(),
            [collection],
            _relations(),
            tmp_path / "production-authority",
            production_ready=True,
        )

    archive = collection / "raw_responses.zip"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(DataReadinessError, match="raw response archive"):
        load_sec_filing_decision_authority(authority.directory, require_production_ready=False)


def test_authority_rejects_unrelated_identity_on_attachment(tmp_path: Path) -> None:
    authority = publish_sec_filing_decision_authority(
        _decisions(),
        [_collection(tmp_path)],
        _relations(),
        tmp_path / "authority",
    )
    altered = _decisions()
    altered.loc[altered["decision_id"].eq("aapl-after"), "ticker"] = "MSFT"
    with pytest.raises(DataReadinessError, match="ticker conflicts"):
        attach_sec_filing_features(altered, authority, require_production_ready=False)


def test_authority_monthly_streaming_memory_contract(tmp_path: Path) -> None:
    collection = _collection(tmp_path)
    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        for ordinal in range(250):
            rows.append(
                {
                    "decision_id": f"2025-{month:02d}-{ordinal:04d}",
                    "security_id": "security:aapl",
                    "ticker": "AAPL",
                    "decision_time_utc": pd.Timestamp(2025, month, (ordinal % 20) + 1, 20, tz="UTC"),
                }
            )
    decisions = tmp_path / "decisions.parquet"
    relations = tmp_path / "relations.parquet"
    pd.DataFrame(rows).to_parquet(decisions, index=False)
    _relations().to_parquet(relations, index=False)

    authority = publish_sec_filing_decision_authority(
        decisions,
        [collection],
        relations,
        tmp_path / "memory-authority",
    )

    assert authority.decision_rows == 3_000
    assert len(authority.partition_records) == 12
    assert max(int(record["rows"]) for record in authority.partition_records) == 250
    assert max(int(record["sessions"]) for record in authority.partition_records) <= 20
