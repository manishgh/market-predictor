from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.features.cross_sectional import (
    RANK_SUFFIX,
    SECTOR_Z_SUFFIX,
    Z_SUFFIX,
    CrossSectionSpec,
    add_cross_sectional_features,
    cross_sectional_feature_names,
)

SPEC = CrossSectionSpec(minimum_cross_section=50, winsorize_quantile=0.0)


def _panel(
    values_by_session: dict[str, list[float]],
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for session, values in values_by_session.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "session": session,
                    "ticker": f"T{index:03d}",
                    "sector": (sectors[index] if sectors else "Tech"),
                    "rsi": value,
                }
            )
    return pd.DataFrame(rows)


def test_a_feature_high_everywhere_carries_no_signal() -> None:
    """If every stock reads the same, none of them stands out."""

    panel = _panel({"2024-01-02": [68.0] * 60})

    out = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)

    # Identical values give zero spread, so the score is undefined rather than
    # a confident zero.
    assert out[f"rsi{Z_SUFFIX}"].isna().all()


def test_the_same_raw_value_scores_differently_on_different_days() -> None:
    """This is the whole point: 68 is strong on a weak day, ordinary on a hot one."""

    panel = _panel(
        {
            "2024-01-02": [68.0] + [40.0] * 59,  # 68 stands far above peers
            "2024-01-03": [68.0] + [90.0] * 59,  # 68 sits far below peers
        }
    )

    out = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)
    first = out[(out["session"] == "2024-01-02") & (out["ticker"] == "T000")]
    second = out[(out["session"] == "2024-01-03") & (out["ticker"] == "T000")]

    assert float(first[f"rsi{Z_SUFFIX}"].iloc[0]) > 5.0
    assert float(second[f"rsi{Z_SUFFIX}"].iloc[0]) < -5.0


def test_scaling_never_reaches_across_sessions() -> None:
    """A later session must not inform an earlier one."""

    early = _panel({"2024-01-02": list(np.linspace(0.0, 1.0, 60))})
    both = _panel(
        {
            "2024-01-02": list(np.linspace(0.0, 1.0, 60)),
            "2024-01-03": list(np.linspace(100.0, 200.0, 60)),
        }
    )

    only_early = add_cross_sectional_features(early, ["rsi"], spec=SPEC)
    with_later = add_cross_sectional_features(both, ["rsi"], spec=SPEC)
    later_added = with_later[with_later["session"] == "2024-01-02"]

    pd.testing.assert_series_equal(
        only_early[f"rsi{Z_SUFFIX}"].reset_index(drop=True),
        later_added[f"rsi{Z_SUFFIX}"].reset_index(drop=True),
    )


def test_rank_is_centred_so_the_sign_carries_meaning() -> None:
    panel = _panel({"2024-01-02": list(np.linspace(0.0, 1.0, 100))})

    out = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)
    ranked = out[f"rsi{RANK_SUFFIX}"]

    assert ranked.min() == pytest.approx(-0.98, abs=0.02)
    assert ranked.max() == pytest.approx(1.0, abs=0.02)
    assert float(ranked.median()) == pytest.approx(0.0, abs=0.02)


def test_rank_ignores_an_outlier_that_would_distort_a_zscore() -> None:
    values = [1.0] * 59 + [10_000.0]
    panel = _panel({"2024-01-02": values})

    out = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)

    # The extreme row is top of the ranking either way, but the ordering of the
    # remaining stocks survives intact under rank and collapses under z-score.
    normal = out[out["ticker"] != "T059"]
    assert normal[f"rsi{Z_SUFFIX}"].abs().max() < 0.2
    assert normal[f"rsi{RANK_SUFFIX}"].max() < 1.0


def test_winsorizing_stops_one_stock_dominating_the_day() -> None:
    values = list(np.linspace(1.0, 2.0, 59)) + [10_000.0]
    panel = _panel({"2024-01-02": values})

    clipped = add_cross_sectional_features(
        panel,
        ["rsi"],
        spec=CrossSectionSpec(minimum_cross_section=50, winsorize_quantile=0.05),
    )
    unclipped = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)

    normal = slice(0, 59)
    assert (
        clipped[f"rsi{Z_SUFFIX}"][normal].abs().max()
        > unclipped[f"rsi{Z_SUFFIX}"][normal].abs().max()
    )


def test_a_cross_section_below_the_minimum_is_left_unscaled() -> None:
    panel = _panel({"2024-01-02": [1.0, 2.0, 3.0]})

    out = add_cross_sectional_features(panel, ["rsi"], spec=SPEC)

    assert out[f"rsi{Z_SUFFIX}"].isna().all()


def test_sector_scaling_keeps_a_sector_rally_from_reading_as_selection() -> None:
    sectors = ["Tech"] * 60 + ["Utilities"] * 60
    values = [90.0] * 60 + [10.0] * 60
    panel = _panel({"2024-01-02": values}, sectors=sectors)

    out = add_cross_sectional_features(
        panel,
        ["rsi"],
        spec=CrossSectionSpec(minimum_cross_section=50, winsorize_quantile=0.0),
    )

    # Across the whole market the two sectors separate completely.
    assert out[f"rsi{Z_SUFFIX}"].abs().max() > 0.9
    # Within each sector every stock is identical, so nothing is selected.
    assert out[f"rsi{SECTOR_Z_SUFFIX}"].isna().all()


def test_emitted_column_names_are_declared_up_front() -> None:
    names = cross_sectional_feature_names(["rsi", "atr"], spec=SPEC)

    assert names == [
        f"rsi{Z_SUFFIX}",
        f"rsi{RANK_SUFFIX}",
        f"atr{Z_SUFFIX}",
        f"atr{RANK_SUFFIX}",
        f"rsi{SECTOR_Z_SUFFIX}",
        f"atr{SECTOR_Z_SUFFIX}",
    ]


def test_spec_rejects_a_cross_section_too_small_to_mean_anything() -> None:
    with pytest.raises(ValueError, match="not the market"):
        CrossSectionSpec(minimum_cross_section=5)


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(DataReadinessError, match="missing feature columns"):
        add_cross_sectional_features(
            pd.DataFrame({"session": ["2024-01-02"], "sector": ["Tech"]}),
            ["rsi"],
            spec=SPEC,
        )


def test_contract_owner_suffixes_and_hashes_are_stable() -> None:
    spec = CrossSectionSpec(
        minimum_cross_section=20,
        winsorize_quantile=0.05,
    )
    restored = pickle.loads(pickle.dumps(spec))
    names = cross_sectional_feature_names(["rsi", "atr"], spec=spec)

    assert CrossSectionSpec.__module__ == (
        "market_predictor.swing.features.cross_sectional"
    )
    assert restored == spec
    assert type(restored).__module__ == (
        "market_predictor.swing.features.cross_sectional"
    )
    assert (Z_SUFFIX, RANK_SUFFIX, SECTOR_Z_SUFFIX) == (
        "_xs_z",
        "_xs_rank",
        "_sector_z",
    )
    assert hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest() == (
        "2700655375ed15afba2c3c96a49c5c794bb4cf455179df7bd49c7f22ffbec45e"
    )
    assert hashlib.sha256(
        json.dumps(names, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == (
        "3cc6ebd5f00ec0a89737fe7468aac1e782706e782ad3b8e619a977b0ac4f9867"
    )


def test_representative_cross_sectional_output_hash_is_stable() -> None:
    spec = CrossSectionSpec(
        minimum_cross_section=20,
        winsorize_quantile=0.05,
    )
    panel = _panel(
        {"2024-01-02": list(np.linspace(0.0, 39.0, 40))},
        sectors=["Tech"] * 20 + ["Utilities"] * 20,
    )
    output = add_cross_sectional_features(panel, ["rsi"], spec=spec)
    payload = output.loc[
        :, cross_sectional_feature_names(["rsi"], spec=spec)
    ].to_json(orient="split", double_precision=15)

    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == (
        "209c3b424677ac8c282439673917fbef6cf2bae3d37c3ffb338496412ba05cec"
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "market_predictor.edge_rebuild.swing_features",
        "market_predictor.edge_rebuild.swing_pipeline_steps",
    ),
)
def test_cross_sectional_contract_has_no_accidental_consumer_aliases(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    assert not hasattr(module, "CrossSectionSpec")
    assert not hasattr(module, "Z_SUFFIX")
    assert not hasattr(module, "RANK_SUFFIX")
    assert not hasattr(module, "SECTOR_Z_SUFFIX")


def test_output_collisions_fail_closed_and_empty_frames_preserve_schema() -> None:
    panel = _panel({"2024-01-02": list(np.linspace(0.0, 1.0, 60))})
    panel[f"rsi{Z_SUFFIX}"] = 0.0

    with pytest.raises(DataReadinessError, match="output columns already exist"):
        add_cross_sectional_features(panel, ["rsi"], spec=SPEC)

    empty = panel.iloc[:0].drop(columns=f"rsi{Z_SUFFIX}")
    output = add_cross_sectional_features(empty, ["rsi"], spec=SPEC)
    pd.testing.assert_frame_equal(output, empty)
