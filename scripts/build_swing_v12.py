import os
import shutil
from pathlib import Path
import pandas as pd
import warnings
from market_predictor.edge_rebuild.swing_pipeline_steps import AdvancedIndicatorsStep

V11_PANEL_DIR = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v11/final/panel")
V12_PANEL_DIR = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final/panel")

def build_v12():
    print(f"Reading V11 panel from {V11_PANEL_DIR}...")
    
    if not V11_PANEL_DIR.exists():
        print(f"Error: {V11_PANEL_DIR} does not exist.")
        return

    # Create V12 directory
    if V12_PANEL_DIR.exists():
        shutil.rmtree(V12_PANEL_DIR)
    V12_PANEL_DIR.mkdir(parents=True)

    # Process each feature profile
    for profile_dir in V11_PANEL_DIR.iterdir():
        if not profile_dir.is_dir() or not profile_dir.name.startswith("feature_profile="):
            continue
            
        profile_name = profile_dir.name
        print(f"Processing {profile_name}...")
        
        # Load the parquet file(s) for this profile
        df = pd.read_parquet(profile_dir)
        print(f"  Loaded {len(df)} rows. Columns include: {[c for c in df.columns if c in ['close', 'volume', 'return_1d', 'atr_pct_14']]}")
        
        # Apply the AdvancedIndicatorsStep
        step = AdvancedIndicatorsStep()
        print("  Applying AdvancedIndicatorsStep transformations...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_df = step.transform(df)
            
        # Verify the features were added
        added = [c for c in ["rsi_14", "macd", "dist_sma_50", "bb_pb"] if c in transformed_df.columns]
        print(f"  Successfully added features: {added}")
        
        # Write to V12 directory
        out_dir = V12_PANEL_DIR / profile_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0000.parquet"
        print(f"  Saving to {out_path}...")
        transformed_df.to_parquet(out_path, index=False)
        
    print("V12 panel creation complete!")

if __name__ == "__main__":
    build_v12()
