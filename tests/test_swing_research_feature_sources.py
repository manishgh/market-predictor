from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets.research_feature_sources import corrected_adjusted_bars, raw_dollar_volume_inputs
from tests.test_swing_corrected_outcomes import publication as publication


def _decisions(fixture: dict[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(fixture["root"] / "data/features/parent/2024-01.parquet").query("security_id == 'common:A'")


def _raw(fixture: dict[str, Any]) -> pd.DataFrame:
    return raw_dollar_volume_inputs(root=fixture["root"], selection=fixture["source"]["selection"],
        decisions=_decisions(fixture), policy=fixture["policy"], policy_sha256=fixture["pin"])


def test_raw_volume_uses_exact_decision_prices_and_close_proxy(publication: dict[str, Any]) -> None:
    result = _raw(publication)
    assert len(result) == 2
    assert result.close.tolist() == [101.0, 102.0]
    assert result.volume.eq(1000.0).all() and result.adjustment.eq("raw").all()
    assert result.available_at_utc.tolist() == list(pd.to_datetime(["2024-01-02T21:15Z", "2024-01-03T21:15Z"]))
    assert result.decision_id.tolist() == _decisions(publication).decision_id.tolist()


def test_invalid_observation_is_absent_not_adjusted_or_imputed(publication: dict[str, Any]) -> None:
    segment = publication["source"]["selection"]["segments"][0]
    path = Path(segment["archive"]) / segment["artifact"]["bars_path"]
    bars = pd.read_parquet(path)
    bars.loc[0, "volume"] = 0.0
    bars.to_parquet(path, index=False)
    segment["artifact"]["bars_sha256"] = file_sha256(path)
    result = _raw(publication)
    assert len(result) == 1
    assert result.decision_id.iloc[0] == _decisions(publication).decision_id.iloc[1]


def test_duplicate_source_ownership_cannot_double_volume(publication: dict[str, Any]) -> None:
    segments = publication["source"]["selection"]["segments"]
    segments.append(segments[0])
    with pytest.raises(DataReadinessError, match="one exact selected segment"):
        _raw(publication)


def _adjusted(fixture: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source = fixture["source"]["selection"]["segments"][0]
    raw = Path(source["archive"]) / source["artifact"]["bars_path"]
    frame = pd.read_parquet(raw)
    frame["adjustment"] = "all"
    path = raw.with_name("adjusted.parquet")
    frame.to_parquet(path, index=False)
    return path, {**source["artifact"], "bars_path": path.name, "bars_sha256": file_sha256(path), "rows": len(frame)}


def test_adjusted_source_reuses_canonical_market_intervals(publication: dict[str, Any]) -> None:
    path, record = _adjusted(publication)
    result = corrected_adjusted_bars(path.parent, record)
    assert result.timeframe.eq("1d").all()
    assert result.availability_policy.eq("market_interval_close").all()
    assert result.bar_start_utc.iloc[0] == pd.Timestamp("2024-01-02T14:30Z")
    assert result.available_at_utc.iloc[0] == pd.Timestamp("2024-01-02T21:15Z")


def test_adjusted_source_wrong_identity_rejected(publication: dict[str, Any]) -> None:
    path, record = _adjusted(publication)
    with pytest.raises(DataReadinessError, match="issuer/feed/basis mismatch"):
        corrected_adjusted_bars(path.parent, {**record, "security_id": "wrong-issuer"})


def test_adjusted_source_numeric_projection_stops_at_initial_fit(
    publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, record = _adjusted(publication)
    original = pd.read_parquet

    def bounded(*args: Any, **kwargs: Any) -> pd.DataFrame:
        assert kwargs["filters"] == [("session_date", ">=", "2018-05-29"), ("session_date", "<=", "2024-05-28")]
        assert "columns" in kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", bounded)
    corrected_adjusted_bars(path.parent, record)
