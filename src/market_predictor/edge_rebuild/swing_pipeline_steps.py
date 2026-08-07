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

class AdvancedIndicatorsStep:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        data = df.sort_values(["security_id", "session_date_et"], kind="stable").copy()
        grouped = data.groupby("security_id", sort=False)
        
        # RSI 14
        delta = grouped["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.groupby(data["security_id"], sort=False).rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(data["security_id"], sort=False).rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
        
        rs = avg_gain / avg_loss
        data["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = grouped["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema_26 = grouped["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data.groupby("security_id", sort=False)["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        
        # Rolling Volatility 20d
        if "return_1d" in data.columns:
            data["volatility_20d"] = grouped["return_1d"].transform(lambda x: x.rolling(window=20, min_periods=1).std())
            
        # SMA 50 and 200 (distance to)
        sma_50 = grouped["close"].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        sma_200 = grouped["close"].transform(lambda x: x.rolling(window=200, min_periods=1).mean())
        data["dist_sma_50"] = (data["close"] - sma_50) / sma_50
        data["dist_sma_200"] = (data["close"] - sma_200) / sma_200
        
        # Bollinger Bands (20, 2)
        sma_20 = grouped["close"].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        std_20 = grouped["close"].transform(lambda x: x.rolling(window=20, min_periods=1).std())
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        data["bb_upper_dist"] = (data["close"] - bb_upper) / bb_upper
        data["bb_lower_dist"] = (data["close"] - bb_lower) / bb_lower
        
        # %b (handling division by zero by replacing zero width with NaNs)
        bb_width = bb_upper - bb_lower
        bb_width_safe = bb_width.replace(0, pd.NA)
        data["bb_pb"] = (data["close"] - bb_lower) / bb_width_safe
        data["bb_pb"] = data["bb_pb"].fillna(0)
            
        return data

