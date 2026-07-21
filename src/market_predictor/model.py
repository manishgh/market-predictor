from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "return_1d",
    "return_5d_past",
    "return_10d_past",
    "return_20d_past",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "atr_pct_14",
    "rsi_14",
    "macd_signal_diff",
    "dist_sma_20",
    "dist_sma_50",
    "sma20_gt_sma50",
    "volume_z20",
    "gap_pct",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "spy_return_1d",
    "spy_return_5d_past",
    "spy_return_10d_past",
    "spy_return_20d_past",
    "spy_realized_vol_20d",
    "spy_volume_z20",
    "spy_gap_pct",
    "sector_return_1d",
    "sector_return_5d_past",
    "sector_return_10d_past",
    "sector_return_20d_past",
    "sector_realized_vol_20d",
    "sector_volume_z20",
    "sector_gap_pct",
    "rel_return_1d_vs_spy",
    "rel_return_5d_vs_spy",
    "rel_return_10d_vs_spy",
    "rel_return_20d_vs_spy",
    "rel_return_1d_vs_sector",
    "rel_return_5d_vs_sector",
    "rel_return_10d_vs_sector",
    "rel_return_20d_vs_sector",
    "news_count",
    "news_count_z30",
    "has_news",
    "sentiment_mean",
    "sentiment_min",
    "sentiment_max",
    "sentiment_pos_frac",
    "sentiment_neg_frac",
    "sentiment_momentum_5d",
    "market_context_news_count",
    "market_context_sentiment_mean",
    "market_context_sentiment_min",
    "market_context_sentiment_max",
    "market_context_sentiment_neg_frac",
    "market_context_sentiment_pos_frac",
    "market_context_news_count_z30",
    "market_context_sentiment_momentum_5d",
    "source_count_alpaca",
    "source_count_reddit",
    "source_count_seeking_alpha",
    "source_count_sec",
    "source_count_finviz",
    "reddit_mentions",
    "reddit_velocity_7d",
    "reddit_newly_trending",
    "reddit_sentiment_mean",
    "reddit_score_sum",
    "reddit_comments_sum",
    "reddit_upvote_ratio_mean",
    "event_count",
    "event_earnings_count",
    "event_analyst_count",
    "event_guidance_count",
    "event_ma_count",
    "event_fda_count",
    "event_contract_count",
    "event_sec_count",
    "event_offering_count",
    "event_insider_count",
    "event_reaction_2h_mean",
    "event_reaction_2h_abs_max",
    "event_reaction_volume_sum",
    "premarket_gap_mean",
    "premarket_day_return_mean",
    "intraday_reaction_2h_mean",
    "intraday_to_close_mean",
    "afterhours_next_open_gap_mean",
    "afterhours_next_day_return_mean",
    "buzz_spike_x_volume_z",
    "sentiment_x_news_attention",
    "catalyst_x_volume_z",
    "reaction_x_sentiment",
    "premarket_gap_x_sentiment",
]


class DateGroupedPurgedWalkForwardSplit:
    """Walk-forward splitter that keeps each timestamp/date group in one fold.

    Row-count embargoes are not sufficient for multi-symbol trading data because
    a single market day can contain hundreds of adjacent rows. This splitter
    treats the supplied ``groups`` values as ordered time buckets and embargoes
    whole buckets.
    """

    def __init__(self, n_splits: int = 5, embargo_groups: int = 1, min_train_size: int = 40) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        if embargo_groups < 0:
            raise ValueError("embargo_groups must be >= 0.")
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups
        self.min_train_size = min_train_size

    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("DateGroupedPurgedWalkForwardSplit requires groups.")
        n_rows = len(x)
        group_series = pd.Series(groups).reset_index(drop=True)
        if len(group_series) != n_rows:
            raise ValueError("groups length must match x length.")
        normalized = self._normalize_groups(group_series)
        ordered_groups = list(pd.Index(normalized.dropna().unique()).sort_values())
        if len(ordered_groups) < self.n_splits + 1:
            return
        fold_size = max(1, len(ordered_groups) // (self.n_splits + 1))
        for split_num in range(self.n_splits):
            test_start_group = min(len(ordered_groups), self.min_train_groups(ordered_groups, fold_size) + split_num * fold_size)
            test_end_group = min(test_start_group + fold_size, len(ordered_groups))
            train_end_group = max(0, test_start_group - self.embargo_groups)
            if test_start_group >= len(ordered_groups) or test_end_group <= test_start_group:
                continue
            train_groups = set(ordered_groups[:train_end_group])
            test_groups = set(ordered_groups[test_start_group:test_end_group])
            train_idx = np.flatnonzero(normalized.isin(train_groups).to_numpy())
            test_idx = np.flatnonzero(normalized.isin(test_groups).to_numpy())
            if len(train_idx) < self.min_train_size or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, x: pd.DataFrame | None = None, y: pd.Series | None = None, groups: object | None = None) -> int:
        if x is None:
            return self.n_splits
        return sum(1 for _ in self.split(x, y, groups))

    def min_train_groups(self, ordered_groups: list[object], fold_size: int) -> int:
        return min(max(1, fold_size), max(1, len(ordered_groups) // 3))

    @staticmethod
    def _normalize_groups(groups: pd.Series) -> pd.Series:
        converted = pd.to_datetime(groups, errors="coerce", utc=True)
        if converted.notna().any():
            return converted.dt.tz_convert(None)
        return groups.astype("string")
