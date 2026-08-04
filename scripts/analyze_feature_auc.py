import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_features import build_swing_feature_rows

def analyze_swing():
    print("\n--- Swing Feature AUC Analysis (On the Fly) ---")
    
    # We load raw bars for a small sample of symbols to run the pipeline
    raw_path = Path("data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/bars/1d/SPY.parquet")
    if not raw_path.exists():
        print("Raw SPY bars not found, falling back to full dataset (which may be missing columns).")
        return
        
    contract = load_strategy_contract(Path("configs/edge_rebuild_strategy_contract.toml"))
    
    # Need stock bars, benchmark bars, memberships
    try:
        spy_bars = pd.read_parquet(raw_path)
        aapl_path = Path("data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3/bars/1d/A.parquet")
        aapl_bars = pd.read_parquet(aapl_path) if aapl_path.exists() else spy_bars.copy()
        
        spy_bars["session_date_et"] = pd.to_datetime(spy_bars["bar_start_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
        aapl_bars["session_date_et"] = pd.to_datetime(aapl_bars["bar_start_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
        
        aapl_bars["ticker"] = "A"
        aapl_bars["security_id"] = "A"
        aapl_bars["sector"] = "Technology"
        
        memberships = pd.DataFrame({
            "ticker": ["A"],
            "security_id": ["A"],
            "sector": ["Technology"],
            "industry": ["Software"],
            "market_cap_bucket": ["mega"],
            "liquidity_bucket": ["high"],
            "primary_benchmark": ["SPY"],
            "source": ["dummy"],
            "universe_snapshot_id": ["dummy_1"],
            "effective_from_utc": [pd.to_datetime("2010-01-01", utc=True)],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.to_datetime("2010-01-01", utc=True)]
        })
        
        print("Materializing Swing features...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_swing_feature_rows(
                stock_bars=aapl_bars,
                benchmark_bars=spy_bars,
                memberships=memberships,
                contract=contract
            )
            
        target = (df["barrier_net_return"] > 0).astype(int)
        print(f"Computed {len(df)} rows.")
        
        SWING_NEW_FEATURES = ["rsi_14", "macd", "dist_sma_50", "dist_sma_200", "bb_upper_dist", "bb_lower_dist"]
        for feature in SWING_NEW_FEATURES:
            if feature in df.columns:
                valid_mask = df[feature].notna() & target.notna()
                y_true = target[valid_mask]
                y_score = df[feature][valid_mask]
                if len(y_true) > 0:
                    try:
                        auc = roc_auc_score(y_true, y_score)
                        predictive_power = max(auc, 1.0 - auc)
                        print(f"  {feature}: {predictive_power:.4f} (Raw: {auc:.4f}) over {len(y_true)} rows")
                    except ValueError as e:
                        pass
            else:
                print(f"  {feature}: Not present in generated frame")
                
    except Exception as e:
        print(f"Failed to generate swing features: {e}")

def analyze_intraday():
    print("\n--- Intraday Feature AUC Analysis (On the Fly) ---")
    print("Intraday pipeline requires 1m bars. Materializing intraday features...")
    # Intraday pipeline is similar
    try:
        from market_predictor.edge_rebuild.pipeline import FeaturePipeline
        from market_predictor.edge_rebuild.intraday_pipeline_steps import IntradayAdvancedIndicatorsStep
        
        raw_path = Path("data/raw/ks4_intraday_one_minute_causal_v2_20260727/bars/2024-08/AAPL.parquet")
        if not raw_path.exists():
            print("1m AAPL bars not found. Cannot evaluate intraday locally without heavy dataset build.")
            return
            
        df = pd.read_parquet(raw_path)
        df["security_id"] = "AAPL"
        
        pipeline = FeaturePipeline([
            IntradayAdvancedIndicatorsStep(contract)
        ])
        
        print("Transforming intraday bars...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed = pipeline.transform(df)
            
        INTRADAY_NEW_FEATURES = ["dist_ema_10", "macd", "distance_to_15m_high"]
        # Intraday doesn't generate targets directly in transform, so we just check feature presence
        print("Features populated in pipeline:")
        for feature in INTRADAY_NEW_FEATURES:
            if feature in processed.columns:
                print(f"  {feature}: Successfully materialized!")
            else:
                print(f"  {feature}: Not present in generated frame")
                
    except Exception as e:
        print(f"Failed to generate intraday features: {e}")

if __name__ == "__main__":
    analyze_swing()
    analyze_intraday()
