"""Parquet string storage may vary; source replay values and other dtypes may not."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_predictor.swing.datasets import issuer_reaction_verification as owner


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = pd.DataFrame({
        "selected_event_id": pd.Series(["earnings-0", pd.NA, "guidance-1"], dtype=pd.StringDtype(storage="python")),
        "missing_reason": pd.Series(["", "unknown_source_coverage", ""], dtype=pd.StringDtype(storage="python")),
        "reaction": np.array([0.0, 0.125, -0.25], dtype=np.float32),
        "available_at_utc": pd.to_datetime(["2020-06-02T20:15:00Z", None, "2020-06-03T20:15:00Z"], utc=True).as_unit("ns"),
    })
    saved = expected.copy(deep=True)
    for name in ("selected_event_id", "missing_reason"):
        saved[name] = saved[name].astype(pd.StringDtype(storage="pyarrow"))
    return saved, expected


def test_replay_accepts_nullable_string_storage_only_and_preserves_expected_input() -> None:
    saved, expected = _rows()
    original = expected.copy(deep=True)
    assert saved.selected_event_id.dtype.storage == "pyarrow"
    assert expected.selected_event_id.dtype.storage == "python"
    assert saved.selected_event_id.iloc[1] is expected.selected_event_id.iloc[1] is pd.NA
    assert saved.reaction.iloc[0] == expected.reaction.iloc[0] == np.float32(0.0)
    assert saved.reaction.dtype == expected.reaction.dtype == np.dtype("float32")
    assert saved.available_at_utc.dtype == expected.available_at_utc.dtype == pd.DatetimeTZDtype(unit="ns", tz="UTC")
    owner._assert_replayed_rows_equal(saved, expected)
    pd.testing.assert_frame_equal(expected, original, check_exact=True)
    assert expected.selected_event_id.dtype.storage == "python"


@pytest.mark.parametrize("difference", [
    "text_value", "string_null_mask", "null_sentinel_kind", "numeric_precision", "clock_unit",
    "row_order", "column_order", "object_string",
])
def test_replay_string_storage_normalization_rejects_other_changes(difference: str) -> None:
    saved, expected = _rows()
    original = expected.copy(deep=True)
    if difference == "text_value":
        saved.loc[0, "selected_event_id"] = "foreign-event"
    elif difference == "string_null_mask":
        saved.loc[1, "selected_event_id"] = "newly-invented-event"
    elif difference == "null_sentinel_kind":
        saved["selected_event_id"] = saved.selected_event_id.astype(pd.StringDtype(storage="pyarrow", na_value=np.nan))
        assert saved.selected_event_id.dtype.na_value is not pd.NA
        assert expected.selected_event_id.dtype.na_value is pd.NA
    elif difference == "numeric_precision":
        saved["reaction"] = saved.reaction.astype("float64")
    elif difference == "clock_unit":
        saved["available_at_utc"] = saved.available_at_utc.dt.as_unit("us")
    elif difference == "row_order":
        saved = saved.iloc[::-1].reset_index(drop=True)
    elif difference == "column_order":
        saved = saved.loc[:, list(reversed(saved.columns))]
    else:
        saved["selected_event_id"] = saved.selected_event_id.astype(object)
    with pytest.raises(AssertionError):
        owner._assert_replayed_rows_equal(saved, expected)
    pd.testing.assert_frame_equal(expected, original, check_exact=True)
