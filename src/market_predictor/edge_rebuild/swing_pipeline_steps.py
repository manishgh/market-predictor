from collections.abc import Sequence

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract


class SetupComponentsStep:
    def __init__(self, benchmark_features: pd.DataFrame):
        self.benchmark_features = benchmark_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon_returns = self.benchmark_features.loc[:, ["ticker", "session_date_et", "return_60d"]]
        spy_rows = horizon_returns.loc[horizon_returns["ticker"].astype(str).str.upper().eq("SPY")]
        if spy_rows.empty:
            raise DataReadinessError("swing residual features require SPY benchmark features")
        spy = spy_rows.rename(columns={"return_60d": "spy_return_60d"}).drop(columns="ticker")
        sector = horizon_returns.rename(columns={"ticker": "primary_benchmark", "return_60d": "sector_return_60d"})
        
        data = df.merge(spy, on="session_date_et", how="left", validate="many_to_one")
        data = data.merge(sector, on=["primary_benchmark", "session_date_et"], how="left", validate="many_to_one")
        
        for window in (20, 60):
            stock = pd.to_numeric(data[f"return_{window}d"], errors="coerce")
            data[f"residual_return_{window}d_vs_spy"] = stock - data[f"spy_return_{window}d"]
            data[f"residual_return_{window}d_vs_sector"] = stock - data[f"sector_return_{window}d"]
            
        data = data.sort_values(["security_id", "session_date_et"], kind="stable")
        grouped = data.groupby("security_id", sort=False)
        data["prior_dist_ema_10"] = grouped["dist_ema_10"].shift(1)
        data["prior_dist_sma_200"] = grouped["dist_sma_200"].shift(1)
        data["dollar_volume"] = data["close"] * data["volume"]
        return data

class TechnicalRelationshipsStep:
    def __init__(self, contract: StrategyContract):
        from market_predictor.swing.features.technical_relationships import relationship_spec_from_contract
        self.spec = relationship_spec_from_contract(
            contract,
            group_columns=("security_id",),
            time_column="session_date_et",
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from market_predictor.swing.features.technical_relationships import add_technical_relationship_features
        return add_technical_relationship_features(df, spec=self.spec)




class CrossSectionalValidationStep:
    def __init__(self, expected_security_ids: Sequence[str] | None = None):
        self.expected_security_ids = expected_security_ids

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from market_predictor.edge_rebuild.swing_features import TECHNICAL_RANKING_FEATURES
        required = {
            "security_id",
            "session_date_et",
            "sector",
            "feature_eligible",
            "daily_bar_count",
            "forward_return",
            *TECHNICAL_RANKING_FEATURES,
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise DataReadinessError(
                f"swing feature rows are missing required columns: {missing}"
            )
        if df.empty:
            raise DataReadinessError("swing feature panel cannot be empty")
        identity = ["security_id", "session_date_et"]
        if bool(df.duplicated(identity).any()):
            raise DataReadinessError(
                "swing feature panel requires one row per security and session"
            )
        if self.expected_security_ids is not None:
            expected = {str(value) for value in self.expected_security_ids}
            observed = set(df["security_id"].astype(str))
            missing_identities = sorted(expected.difference(observed))
            unexpected_identities = sorted(observed.difference(expected))
            if missing_identities:
                raise DataReadinessError(
                    "population-wide swing scaling is missing expected securities: "
                    f"{missing_identities[:10]}"
                )
            if unexpected_identities:
                raise DataReadinessError(
                    "population-wide swing scaling contains unexpected securities: "
                    f"{unexpected_identities[:10]}"
                )
        return df


class SectorRelativeScalingStep:
    def __init__(self, contract: StrategyContract):
        self.contract = contract

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from market_predictor.edge_rebuild.swing_features import (
            CATALYST_RANKING_FEATURES,
            TECHNICAL_RANKING_FEATURES,
            _cross_section_spec,
        )
        from market_predictor.swing.features import (
            cross_sectional as swing_cross_sectional,
        )
        
        data = df.copy()
        profiles = set(data["feature_profile"].astype(str))
        if len(profiles) != 1:
            raise DataReadinessError(
                f"swing feature panel mixes feature profiles: {sorted(profiles)}"
            )
        ranking_inputs = list(TECHNICAL_RANKING_FEATURES)
        if profiles == {"catalyst_full"}:
            missing_catalyst = sorted(
                set(CATALYST_RANKING_FEATURES).difference(data.columns)
            )
            if missing_catalyst:
                raise DataReadinessError(
                    "catalyst swing rows are missing ranking features: "
                    f"{missing_catalyst}"
                )
            ranking_inputs.extend(CATALYST_RANKING_FEATURES)
        spec = _cross_section_spec(self.contract)
        transformed_names = swing_cross_sectional.cross_sectional_feature_names(
            ranking_inputs,
            spec=spec,
        )
        eligible = (
            data["feature_eligible"].fillna(False).astype(bool)
            & data["daily_bar_count"].ge(self.contract.swing.minimum_warmup_sessions)
        )
        transformed = swing_cross_sectional.add_cross_sectional_features(
            data.loc[eligible],
            ranking_inputs,
            spec=spec,
            timestamp_column="session_date_et",
            sector_column="sector",
        )
        transformed_block = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=transformed_names,
            dtype="float32",
        )
        if not transformed.empty:
            transformed_block.loc[eligible, :] = transformed.loc[
                :, transformed_names
            ].to_numpy(dtype=np.float32)
        data = pd.concat([data, transformed_block], axis=1)

        sector_peer_count = pd.Series(0, index=data.index, dtype="int32")
        if bool(eligible.any()):
            eligible_peers = data.loc[eligible, ["session_date_et", "sector"]]
            sector_peer_count.loc[eligible] = (
                eligible_peers.groupby(
                    ["session_date_et", "sector"],
                    sort=False,
                )["sector"]
                .transform("size")
                .to_numpy(dtype="int32")
            )
        sector_rank_eligible = sector_peer_count.ge(
            self.contract.labels.minimum_cross_section_for_ranking
        )
        data["sector_peer_count"] = sector_peer_count
        data["sector_rank_eligible"] = sector_rank_eligible
        data["sector_rank_target_met"] = sector_peer_count.ge(
            self.contract.labels.swing_target_cross_section_for_ranking
        )
        return data


class CrossSectionalRankStep:
    def __init__(self, contract: StrategyContract):
        self.contract = contract

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from market_predictor.edge_rebuild.labeling import apply_cross_sectional_rank
        data = df.copy()
        eligible = (
            data["feature_eligible"].fillna(False).astype(bool)
            & data["daily_bar_count"].ge(self.contract.swing.minimum_warmup_sessions)
        )
        sector_peer_count = data["sector_peer_count"]
        sector_rank_eligible = data["sector_rank_eligible"]
        
        rank_eligible = (
            eligible
            & sector_rank_eligible
            & data["barrier_label"].notna()
            & data["forward_return"].notna()
        )
        ranked = apply_cross_sectional_rank(
            data.loc[
                rank_eligible,
                ["session_date_et", "sector", "forward_return"],
            ].rename(columns={"session_date_et": "session"}),
            top_quantile=self.contract.labels.rank_top_quantile,
            bottom_quantile=self.contract.labels.rank_bottom_quantile,
            within_sector=self.contract.labels.rank_within_sector,
            minimum_cross_section=self.contract.labels.minimum_cross_section_for_ranking,
        )
        rank_label = pd.Series(pd.NA, index=data.index, dtype="Int64")
        rank_label.loc[rank_eligible] = ranked["rank_label"].to_numpy()
        rank_percentile = pd.Series(np.nan, index=data.index, dtype="float32")
        rank_percentile.loc[rank_eligible] = ranked["rank_percentile"].to_numpy(
            dtype=np.float32
        )
        ranking_group_size = pd.Series(pd.NA, index=data.index, dtype="Int32")
        ranking_group_size.loc[rank_eligible] = ranked[
            "ranking_group_size"
        ].to_numpy(dtype="int32")
        ranking_reliability_weight = pd.Series(
            np.nan,
            index=data.index,
            dtype="float32",
        )
        ranking_reliability_weight.loc[sector_rank_eligible] = np.minimum(
            sector_peer_count.loc[sector_rank_eligible].to_numpy(dtype="float32")
            / float(self.contract.labels.swing_target_cross_section_for_ranking),
            1.0,
        )
        data["rank_label"] = rank_label
        data["rank_percentile"] = rank_percentile
        data["ranking_group_size"] = ranking_group_size
        data["ranking_reliability_weight"] = ranking_reliability_weight
        data["cross_section_eligible"] = sector_rank_eligible
        return data.sort_values(
            ["session_date_et", "security_id"],
            kind="stable",
        ).reset_index(drop=True)

