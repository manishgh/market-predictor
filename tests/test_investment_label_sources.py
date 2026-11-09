"""Raw outcome source hashes and actual capture clocks survive horizon expansion."""
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.label_sources import InvestmentSourceReader


def reader(tmp_path: Path, *, capture: str = "2026-11-07T00:00:00Z") -> InvestmentSourceReader:
    path = tmp_path / "raw.parquet"
    pd.DataFrame({"security_id": ["issuer:test"], "ticker": ["TEST"], "session_date": ["2025-05-27"],
        "bar_start_utc": ["2025-05-27T04:00:00Z"], "open": [100.0], "high": [111.0], "low": [99.0],
        "close": [110.0], "volume": [100], "source": ["alpaca"], "timeframe": ["1Day"],
        "price_feed": ["sip"], "adjustment": ["raw"], "ingested_at_utc": [capture]}).to_parquet(path)
    selection = {"segments": [{"archive": str(tmp_path), "first_session": "2025-05-27", "last_session": "2025-05-27",
        "artifact": {"security_id": "issuer:test", "ticker": "TEST", "provider_symbol": "TEST", "role": "stock",
            "bars_path": "raw.parquet", "bars_sha256": file_sha256(path)}}]}
    return InvestmentSourceReader(tmp_path, selection, "b" * 64)


def test_2025_raw_prices_keep_actual_2026_capture(tmp_path: Path) -> None:
    source = reader(tmp_path)
    observed = source.read("issuer:test", "TEST", (date(2025, 5, 27),), benchmark=False)
    assert not observed.missing_reasons
    assert observed.evidence[date(2025, 5, 27)].available_at == datetime(2026, 11, 7, tzinfo=UTC)
    assert observed.evidence[date(2025, 5, 27)].artifact_sha256 == file_sha256(tmp_path / "raw.parquet")
    source.recheck()
    (tmp_path / "raw.parquet").write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="changed"):
        source.recheck()


def test_capture_before_economic_observation_is_rejected(tmp_path: Path) -> None:
    source = reader(tmp_path, capture="2025-05-27T19:00:00Z")
    with pytest.raises(DataReadinessError, match="availability mismatch"):
        source.read("issuer:test", "TEST", (date(2025, 5, 27),), benchmark=False)


def test_missing_future_segment_is_explicit(tmp_path: Path) -> None:
    observed = reader(tmp_path).read("issuer:test", "TEST", (date(2025, 5, 28),), benchmark=False)
    assert observed.missing_reasons == ("missing_or_cross_segment_raw_binding",)


def composed_reader(tmp_path: Path) -> InvestmentSourceReader:
    import copy

    source = reader(tmp_path)
    segment = copy.deepcopy(source.selection["segments"][0])
    segment.update(first_session="2025-05-28", last_session="2025-05-28")
    frame = pd.read_parquet(tmp_path / "raw.parquet")
    frame["session_date"] = "2025-05-28"
    frame["bar_start_utc"] = "2025-05-28T04:00:00Z"
    frame["ingested_at_utc"] = "2026-11-09T00:00:00Z"
    frame.to_parquet(tmp_path / "extension.parquet")
    segment["artifact"].update(bars_path="extension.parquet", bars_sha256=file_sha256(tmp_path / "extension.parquet"))
    source.selection["segments"].append(segment)
    return source


def test_adjacent_segments_preserve_each_original_source_clock(tmp_path: Path) -> None:
    source = composed_reader(tmp_path)
    observed = source.read("issuer:test", "TEST", (date(2025, 5, 27), date(2025, 5, 28)), benchmark=False)
    assert not observed.missing_reasons
    assert len(observed.frame) == 2
    assert observed.evidence[date(2025, 5, 27)].reference == "raw.parquet"
    assert observed.evidence[date(2025, 5, 28)].reference == "extension.parquet"
    assert observed.evidence[date(2025, 5, 28)].available_at == datetime(2026, 11, 9, tzinfo=UTC)
    source.recheck()


def test_overlap_never_chooses_a_preferred_segment_even_for_identical_bytes(tmp_path: Path) -> None:
    source = reader(tmp_path)
    source.selection["segments"].append(dict(source.selection["segments"][0]))
    with pytest.raises(DataReadinessError, match="overlap"):
        source.read("issuer:test", "TEST", (date(2025, 5, 27),), benchmark=False)


def test_quarantined_stock_tail_cannot_authorize_a_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = composed_reader(tmp_path)
    source.selection["segments"][1]["source_admission"] = "quarantined_unreviewed_provider_candidate"
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("quarantine must precede price read"))
    observed = source.read("issuer:test", "TEST", (date(2025, 5, 27), date(2025, 5, 28)), benchmark=False)
    assert observed.missing_reasons == ("raw_stock_extension_identity_unreviewed",)
    assert not observed.evidence and observed.frame.empty


def test_cross_segment_provider_change_remains_unresolved(tmp_path: Path) -> None:
    source = composed_reader(tmp_path)
    source.selection["segments"][1]["artifact"]["provider_symbol"] = "DIFFERENT"
    observed = source.read("issuer:test", "TEST", (date(2025, 5, 27), date(2025, 5, 28)), benchmark=False)
    assert observed.missing_reasons == ("raw_security_symbol_or_role_differs",)


def test_extension_tamper_is_fatal(tmp_path: Path) -> None:
    source = composed_reader(tmp_path)
    (tmp_path / "extension.parquet").write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="hash differs"):
        source.read("issuer:test", "TEST", (date(2025, 5, 27), date(2025, 5, 28)), benchmark=False)


def test_requested_gap_preserves_exact_present_day_clocks(tmp_path: Path) -> None:
    source = composed_reader(tmp_path)
    source.selection["segments"][1]["last_session"] = "2025-05-29"
    observed = source.read("issuer:test", "TEST", (date(2025, 5, 27), date(2025, 5, 28), date(2025, 5, 29)), benchmark=False)
    assert observed.missing_reasons == ("raw_observation_unavailable:2025-05-29",)
    assert set(observed.evidence) == {date(2025, 5, 27), date(2025, 5, 28)}
    assert observed.evidence[date(2025, 5, 27)].available_at == datetime(2026, 11, 7, tzinfo=UTC)
    assert observed.evidence[date(2025, 5, 28)].available_at == datetime(2026, 11, 9, tzinfo=UTC)
