import pandas as pd
from pathlib import Path
from market_predictor.edge_rebuild.cross_sectional import add_cross_sectional_features, CrossSectionSpec
import warnings

def add_xs_features():
    v12_dir = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final/panel")
    
    features_to_scale = [
        "bb_lower_dist", "bb_pb", "bb_upper_dist", "dist_sma_50", 
        "macd_hist", "macd", "macd_signal", "rsi_14", "volatility_20d"
    ]
    
    spec = CrossSectionSpec()
    
    for profile in ["catalyst_full", "technical_market"]:
        profile_dir = v12_dir / f"feature_profile={profile}"
        if not profile_dir.exists():
            continue
            
        print(f"Processing {profile}...")
        
        for month_dir in profile_dir.glob("month=*"):
            if not month_dir.is_dir():
                continue
                
            pqt_file = month_dir / "part.parquet"
            if not pqt_file.exists():
                continue
                
            df = pd.read_parquet(pqt_file)
            
            # Check if we already added them
            if f"{features_to_scale[0]}_xs_z" in df.columns:
                print(f"  {month_dir.name} already has xs features. Skipping.")
                continue
                
            # If any of the base features are missing (unlikely, but just in case)
            missing = [f for f in features_to_scale if f not in df.columns]
            if missing:
                print(f"  Warning: {month_dir.name} missing base features: {missing}. Cannot compute xs.")
                continue
                
            # Add cross sectional features
            # Note: Cross sectional logic groups by timestamp. So doing it per month is FINE 
            # because the grouping is per session!
            df = add_cross_sectional_features(
                df,
                features_to_scale,
                spec=spec,
                timestamp_column="session_date_et",
                sector_column="sector"
            )
            
            # Save it back
            df.to_parquet(pqt_file, index=False)
            print(f"  Processed {month_dir.name} -> added {len([c for c in df.columns if '_xs_z' in c])} total xs_z columns")
            
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        add_xs_features()
