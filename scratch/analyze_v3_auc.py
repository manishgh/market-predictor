import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings

def analyze_v3():
    print("Reading Intraday V3 panel to calculate AUCs...")
    v3_dir = Path("data/features/edge_rebuild_intraday_dataset_causal_20260804_v3/partitions")
    
    # Just take the first few partitions to get a sample, or we can load all.
    dfs = []
    partitions = list(v3_dir.glob("session_month_et=*"))
    for p in partitions[:3]:  # Use 3 months of data to get a solid sample
        for f in p.glob("*.parquet"):
            df = pd.read_parquet(f, columns=["macd", "ema_10_distance", "sma_10_distance", "target_hit"])
            dfs.append(df)
            
    if not dfs:
        print("No data found!")
        return
        
    full_df = pd.concat(dfs).dropna()
    print(f"Loaded {len(full_df)} rows.")
    
    # Ensure target_hit is numeric (it usually is boolean/int)
    full_df["target_hit"] = full_df["target_hit"].astype(int)
    
    for feature in ["macd", "ema_10_distance", "sma_10_distance"]:
        if feature not in full_df.columns:
            print(f"Feature {feature} not found.")
            continue
            
        try:
            auc = roc_auc_score(full_df["target_hit"], full_df[feature])
            # If auc < 0.5, the inverse relation is stronger
            display_auc = max(auc, 1 - auc)
            print(f"  {feature}: {display_auc:.4f} (Raw: {auc:.4f})")
        except Exception as e:
            print(f"  Failed on {feature}: {e}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyze_v3()
