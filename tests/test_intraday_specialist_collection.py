from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.specialist_collection import (
    SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA,
    _unit_bundle_fingerprint,
    build_intraday_specialist_acquisition_units,
    collect_intraday_specialist_one_minute,
    verify_intraday_specialist_acquisition_units,
)
from market_predictor.intraday.specialist_contracts import (
    intraday_specialist_policy_identity,
)
from market_predictor.intraday.specialist_dataset import (
    SPECIALIST_COLLECTION_PLAN_SCHEMA,
    _collection_plan_fingerprint,
)
from market_predictor.sources.alpaca import AlpacaBarsPage
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "intraday_specialist_research.toml"


class IntradaySpecialistCollectionTests(unittest.TestCase):
    def test_unit_planner_batches_normal_and_early_close_sessions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan"
            windows_path = plan / "collection_windows" / "2026-06.parquet"
            windows_path.parent.mkdir(parents=True)
            tickers = [f"T{index:03d}" for index in range(25)]
            windows = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "session_date_et": session,
                        "price_feed": "sip",
                        "adjustment": "all",
                        "timeframe": "1m",
                    }
                    for session in (
                        pd.Timestamp("2026-06-01").date(),
                        pd.Timestamp("2026-11-27").date(),
                    )
                    for ticker in tickers
                ]
            )
            windows.to_parquet(windows_path, index=False)
            file_record = {
                "path": windows_path.relative_to(plan).as_posix(),
                "sha256": file_sha256(windows_path),
                "bytes": windows_path.stat().st_size,
                "rows": len(windows),
            }
            policy = intraday_specialist_policy_identity(POLICY)
            plan_fingerprint = _collection_plan_fingerprint(
                files=[file_record],
                setup_bundle_fingerprint="s" * 64,
                policy_sha256=policy["policy_sha256"],
            )
            (plan / "_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": SPECIALIST_COLLECTION_PLAN_SCHEMA,
                        "plan_fingerprint": plan_fingerprint,
                        "setup_bundle": {
                            "bundle_fingerprint": "s" * 64
                        },
                        "policy": policy,
                        "files": [file_record],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "units"

            report = build_intraday_specialist_acquisition_units(
                collection_plan_directory=plan,
                policy_path=POLICY,
                output_directory=output,
            )

            self.assertEqual(report["summary"]["units"], 3)
            self.assertEqual(
                report["summary"]["maximum_expected_rows"],
                15_000,
            )
            verified = verify_intraday_specialist_acquisition_units(
                output
            )
            self.assertEqual(
                verified["unit_bundle_fingerprint"],
                report["unit_bundle_fingerprint"],
            )

    def test_collector_publishes_canonical_sip_unit_and_final_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = _write_unit_bundle(root / "units")
            output = root / "collection"
            fake = _FakeAlpacaSource()

            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=units,
                policy_path=POLICY,
                output_directory=output,
                source_factory=lambda: fake,
            )

            self.assertEqual(report["status"], "transport_complete")
            self.assertEqual(report["coverage_status"], "not_evaluated")
            self.assertFalse(report["model_data_ready"])
            self.assertEqual(report["completed_units"], 1)
            self.assertEqual(report["total_rows"], 4)
            artifact = report["artifacts"][0]
            bars = pd.read_parquet(artifact["path"])
            self.assertEqual(set(bars["ticker"]), {"AAA", "BBB"})
            self.assertEqual(set(bars["price_feed"]), {"sip"})
            self.assertEqual(set(bars["adjustment"]), {"all"})
            self.assertTrue((output / "_manifest.json").exists())
            raw_page = Path(artifact["pages"][0]["raw_page_path"])
            self.assertEqual(
                artifact["pages"][0]["raw_page_sha256"],
                file_sha256(raw_page),
            )
            payload = json.loads(gzip.decompress(raw_page.read_bytes()))
            self.assertEqual(set(payload["bars"]), {"AAA", "BBB"})
            self.assertEqual(
                fake.requested_ends,
                [datetime.fromisoformat("2026-06-01T13:31:59.999999+00:00")],
            )

    def test_collector_resumes_integrity_checked_unit_without_network(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = _write_unit_bundle(root / "units")
            output = root / "collection"
            first = collect_intraday_specialist_one_minute(
                acquisition_units_directory=units,
                policy_path=POLICY,
                output_directory=output,
                source_factory=_FakeAlpacaSource,
            )
            (output / "_manifest.json").unlink()

            def unexpected_source() -> _FakeAlpacaSource:
                raise AssertionError("resume must not call Alpaca")

            resumed = collect_intraday_specialist_one_minute(
                acquisition_units_directory=units,
                policy_path=POLICY,
                output_directory=output,
                source_factory=unexpected_source,
            )

            self.assertEqual(first["total_rows"], resumed["total_rows"])
            self.assertEqual(resumed["resumed_units"], 1)
            self.assertEqual(resumed["status"], "transport_complete")

    def test_collector_rejects_repeated_pagination_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=_write_unit_bundle(
                    root / "units"
                ),
                policy_path=POLICY,
                output_directory=root / "collection",
                source_factory=_RepeatingTokenAlpacaSource,
            )

            self.assertEqual(report["status"], "transport_incomplete")
            self.assertEqual(report["completed_units"], 0)
            self.assertEqual(len(report["failed_units"]), 1)
            error = next(iter(report["failed_units"].values()))
            self.assertIn("repeated a page token", error)

    def test_collector_rejects_final_page_above_row_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=_write_unit_bundle(
                    root / "units"
                ),
                policy_path=POLICY,
                output_directory=root / "collection",
                source_factory=_OversizedAlpacaSource,
            )

            self.assertEqual(report["status"], "transport_incomplete")
            error = next(iter(report["failed_units"].values()))
            self.assertIn("exceeded its bounded row budget", error)

    def test_collector_bounds_unique_empty_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=_write_unit_bundle(
                    root / "units"
                ),
                policy_path=POLICY,
                output_directory=root / "collection",
                source_factory=_UnboundedTokenAlpacaSource,
            )

            self.assertEqual(report["status"], "transport_incomplete")
            error = next(iter(report["failed_units"].values()))
            self.assertIn("exceeded the page budget", error)

    def test_collector_quarantines_orphaned_unit_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = _write_unit_bundle(root / "units")
            output = root / "collection"
            orphan = output / "bars" / "2026-06" / f"{'u' * 64}.parquet"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"interrupted")

            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=units,
                policy_path=POLICY,
                output_directory=output,
                source_factory=_FakeAlpacaSource,
            )

            self.assertEqual(report["status"], "transport_complete")
            quarantined = list(
                (orphan.parent / "_quarantine").glob("*.orphan")
            )
            self.assertEqual(len(quarantined), 1)

    def test_resume_rejects_modified_collected_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = _write_unit_bundle(root / "units")
            output = root / "collection"
            report = collect_intraday_specialist_one_minute(
                acquisition_units_directory=units,
                policy_path=POLICY,
                output_directory=output,
                source_factory=_FakeAlpacaSource,
            )
            artifact = Path(report["artifacts"][0]["path"])
            bars = pd.read_parquet(artifact)
            bars.loc[0, "close"] = 999.0
            bars.to_parquet(artifact, index=False)
            (output / "_manifest.json").unlink()

            with self.assertRaisesRegex(
                DataReadinessError,
                "integrity failed",
            ):
                collect_intraday_specialist_one_minute(
                    acquisition_units_directory=units,
                    policy_path=POLICY,
                    output_directory=output,
                    source_factory=_FakeAlpacaSource,
                )


class _FakeAlpacaSource:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=30)
        self.requested_ends: list[datetime] = []

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **_: object,
    ) -> AlpacaBarsPage:
        self.requested_ends.append(end)
        timestamps = [
            pd.Timestamp(start),
            pd.Timestamp(start) + pd.Timedelta(minutes=1),
        ]
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars={
                symbol: tuple(
                    {
                        "t": timestamp.isoformat(),
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1000,
                    }
                    for timestamp in timestamps
                )
                for symbol in symbols
            },
            response_headers={"X-RateLimit-Remaining": "100"},
        )


class _RepeatingTokenAlpacaSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        page = super().fetch_bars_page(symbols, start, end, **kwargs)
        return AlpacaBarsPage(
            request_page_token=kwargs.get("page_token"),
            next_page_token="repeat",
            bars=page.bars,
            response_headers=page.response_headers,
        )


class _OversizedAlpacaSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **_: object,
    ) -> AlpacaBarsPage:
        timestamps = [
            pd.Timestamp(start) + pd.Timedelta(seconds=index)
            for index in range(9)
        ]
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars={
                symbols[0]: tuple(
                    {
                        "t": timestamp.isoformat(),
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1000,
                    }
                    for timestamp in timestamps
                )
            },
            response_headers={},
        )


class _UnboundedTokenAlpacaSource(_FakeAlpacaSource):
    def __init__(self) -> None:
        super().__init__()
        self.page = 0

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        self.page += 1
        return AlpacaBarsPage(
            request_page_token=kwargs.get("page_token"),
            next_page_token=f"page-{self.page}",
            bars={},
            response_headers={},
        )


def _write_unit_bundle(directory: Path) -> Path:
    directory.mkdir()
    units_path = directory / "units" / "2026-06.parquet"
    units_path.parent.mkdir()
    policy = intraday_specialist_policy_identity(POLICY)
    unit = pd.DataFrame(
        [
            {
                "unit_id": "u" * 64,
                "session_date_et": pd.Timestamp("2026-06-01").date(),
                "requested_start_utc": pd.Timestamp(
                    "2026-06-01T13:30:00Z"
                ),
                "requested_end_utc": pd.Timestamp(
                    "2026-06-01T13:32:00Z"
                ),
                "asof_date": pd.Timestamp("2026-06-01").date(),
                "canonical_symbols_json": '["AAA","BBB"]',
                "provider_symbols_json": '["AAA","BBB"]',
                "provider_to_canonical_json": '{"AAA":"AAA","BBB":"BBB"}',
                "symbol_count": 2,
                "session_minutes": 2,
                "maximum_expected_rows": 4,
                "timeframe": "1Min",
                "price_feed": "sip",
                "adjustment": "all",
                "sort": "asc",
                "limit": 10_000,
                "collection_plan_fingerprint": "p" * 64,
            }
        ]
    )
    unit.to_parquet(units_path, index=False)
    file_record = {
        "path": units_path.relative_to(directory).as_posix(),
        "sha256": file_sha256(units_path),
        "bytes": units_path.stat().st_size,
        "rows": 1,
    }
    fingerprint = _unit_bundle_fingerprint(
        files=[file_record],
        collection_plan_fingerprint="p" * 64,
        policy_sha256=policy["policy_sha256"],
    )
    (directory / "_manifest.json").write_text(
        json.dumps(
            {
                "schema": SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA,
                "unit_bundle_fingerprint": fingerprint,
                "collection_plan": {"plan_fingerprint": "p" * 64},
                "policy": policy,
                "files": [file_record],
            }
        ),
        encoding="utf-8",
    )
    return directory


if __name__ == "__main__":
    unittest.main()
