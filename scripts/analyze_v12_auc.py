import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

def analyze_swing_v12():
    print("\n--- Swing Feature AUC Analysis (V12 Full Panel) ---")
    
    panel_path = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final/panel/feature_profile=technical_market")
    if not panel_path.exists():
        print(f"V12 panel not found at {panel_path}")
        return
        
    print(f"Loading full V12 panel from {panel_path}...")
    df = pd.read_parquet(panel_path)
    
    target = (df["barrier_net_return"] > 0).astype(int)
    print(f"Loaded {len(df)} rows.")
    
    SWING_NEW_FEATURES = ["rsi_14", "macd", "dist_sma_50", "dist_sma_200", "bb_upper_dist", "bb_lower_dist", "bb_pb"]
    
    print("\nFeature Predictive Power (Max of AUC or 1-AUC):")
    for feature in SWING_NEW_FEATURES:
        if feature in df.columns:
            valid_mask = df[feature].notna() & target.notna()
            y_true = target[valid_mask]
            y_score = df[feature][valid_mask]
            if len(y_true) > 0:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    predictive_power = max(auc, 1.0 - auc)
                    print(f"  {feature}: {predictive_power:.4f} (Raw AUC: {auc:.4f}) over {len(y_true):,} rows")
                except ValueError as e:
                    pass
        else:
            print(f"  {feature}: Not present in generated frame")

if __name__ == "__main__":
    analyze_swing_v12()
