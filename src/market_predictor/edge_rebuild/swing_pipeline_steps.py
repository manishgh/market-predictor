import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError


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
        from market_predictor.edge_rebuild.technical_relationships import relationship_spec_from_contract
        self.spec = relationship_spec_from_contract(
            contract,
            group_columns=("security_id",),
            time_column="session_date_et",
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from market_predictor.edge_rebuild.technical_relationships import add_technical_relationship_features
        return add_technical_relationship_features(df, spec=self.spec)
