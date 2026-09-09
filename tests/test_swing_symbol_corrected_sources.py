from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.official_documents import OfficialDocument
from market_predictor.swing.datasets.symbol_corrected_sources import _segment, publish_or_verify_symbol_corrected_sources


def test_fiserv_document_host_is_exact() -> None:
    from pydantic import ValidationError

    kwargs = {"document_id": "fiserv_completion", "purpose": "Official completed listing transfer evidence", "expected_media": "html"}
    OfficialDocument.model_validate({**kwargs, "url": "https://investors.fiserv.com/news-releases/completion"})
    with pytest.raises(ValidationError):
        OfficialDocument.model_validate({**kwargs, "url": "https://investors.fiserv.com.example.com/news-releases/completion"})


def test_issuer_transport_uses_library_identity_without_changing_sec_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from market_predictor.sources import official_documents as module
    from market_predictor.sources.http import HttpClient

    sec = HttpClient(user_agent="test SEC contact identity")
    monkeypatch.setattr(module, "SecSource", lambda settings: SimpleNamespace(client=sec))
    source = module.OfficialDocumentSource(Mock())
    try:
        assert source.sec_client.session.headers["User-Agent"] == "test SEC contact identity"
        assert source.other_client.session.headers["User-Agent"] == module.requests.utils.default_user_agent()
    finally:
        source.close()


def _bars(tmp_path: Path) -> dict[str, object]:
    pd.DataFrame({
        "security_id": ["cik:0000798354"] * 2, "session_date": ["2023-06-07", "2023-06-08"],
        "bar_start_utc": pd.to_datetime(["2023-06-07T04:00:00Z", "2023-06-08T04:00:00Z"]),
        "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
        "close": [101.0, 102.0], "volume": [10000, 11000],
        "ingested_at_utc": pd.to_datetime(["2026-09-09T12:00:00Z"] * 2),
    }).to_parquet(tmp_path / "bars.parquet", index=False)
    return {"bars_path": "bars.parquet", "bars_sha256": file_sha256(tmp_path / "bars.parquet")}


def test_corrected_observations_keep_raw_midnight_and_actual_retrieval(tmp_path: Path) -> None:
    from datetime import date

    artifact = _bars(tmp_path)
    result = _segment(tmp_path, artifact, date(2023, 6, 7), date(2023, 6, 8), replacement=True)
    assert result["rows"] == 2 and result["missing_sessions"] == [] and result["invalid_observations"] == 0
    assert file_sha256(tmp_path / "bars.parquet") == artifact["bars_sha256"]
    assert pd.read_parquet(tmp_path / "bars.parquet").bar_start_utc.dt.hour.eq(4).all()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "non_session", "zero_volume", "bad_high", "future_clock"])
def test_corrected_observations_fail_closed(tmp_path: Path, mutation: str) -> None:
    from datetime import date

    artifact = _bars(tmp_path)
    frame = pd.read_parquet(tmp_path / "bars.parquet")
    if mutation == "missing":
        frame = frame.iloc[:1]
    elif mutation == "duplicate":
        frame.loc[1, "session_date"] = "2023-06-07"
    elif mutation == "non_session":
        frame.loc[1, "session_date"] = "2023-06-10"
    elif mutation == "zero_volume":
        frame.loc[1, "volume"] = 0
    elif mutation == "bad_high":
        frame.loc[1, "high"] = 90.0
    else:
        frame["ingested_at_utc"] = pd.to_datetime(["2023-06-07T12:00:00Z"] * 2)
    frame.to_parquet(tmp_path / "bars.parquet", index=False)
    artifact["bars_sha256"] = file_sha256(tmp_path / "bars.parquet")
    with pytest.raises(DataReadinessError):
        _segment(tmp_path, artifact, date(2023, 6, 7), date(2023, 6, 8), replacement=True)


def test_unresolved_parent_gap_is_explicit_not_imputed(tmp_path: Path) -> None:
    from datetime import date

    result = _segment(tmp_path, _bars(tmp_path), date(2023, 6, 7), date(2023, 6, 9), replacement=False)
    assert result["rows"] == 2
    assert result["missing_sessions"] == ["2023-06-09"]


def test_selected_source_replay_requires_external_pin_and_identical_content(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    result = {"status": "test_only", "label_eligible": False}
    publish_or_verify_symbol_corrected_sources(path, result)
    digest = file_sha256(path)
    publish_or_verify_symbol_corrected_sources(path, result, expected_sha256=digest)
    with pytest.raises(DataReadinessError):
        publish_or_verify_symbol_corrected_sources(path, result)
    with pytest.raises(DataReadinessError):
        publish_or_verify_symbol_corrected_sources(path, {**result, "label_eligible": True}, expected_sha256=digest)
    with pytest.raises(DataReadinessError):
        publish_or_verify_symbol_corrected_sources(path, result, expected_sha256="0" * 64)
