import shutil
import warnings
from pathlib import Path

import pandas as pd
from market_predictor.edge_rebuild.intraday_pipeline_steps import IntradayAdvancedIndicatorsStep

from market_predictor.modeling.strategy_contract import load_strategy_contract

V2_PANEL_DIR = Path("data/features/edge_rebuild_intraday_dataset_causal_20260802_v2/partitions")
V3_PANEL_DIR = Path("data/features/edge_rebuild_intraday_dataset_causal_20260804_v3/partitions")

def build_intraday_v3():
    print(f"Reading Intraday V2 panel from {V2_PANEL_DIR}...")
    
    if not V2_PANEL_DIR.exists():
        print(f"Error: {V2_PANEL_DIR} does not exist.")
        return

    # Create V3 directory
    if V3_PANEL_DIR.exists():
        shutil.rmtree(V3_PANEL_DIR)
    V3_PANEL_DIR.mkdir(parents=True)
    
    contract = load_strategy_contract(Path("configs/edge_rebuild_strategy_contract.toml"))

    # Process each partition
    for partition_dir in V2_PANEL_DIR.iterdir():
        if not partition_dir.is_dir() or not partition_dir.name.startswith("session_month_et="):
            continue
            
        partition_name = partition_dir.name
        print(f"Processing {partition_name}...")
        
        # Load the parquet file(s) for this partition
        try:
            df = pd.read_parquet(partition_dir)
        except Exception as e:
            print(f"  Failed to read {partition_dir}: {e}")
            continue
            
        print(f"  Loaded {len(df)} rows. Applying IntradayAdvancedIndicatorsStep...")
        
        # Apply the AdvancedIndicatorsStep
        step = IntradayAdvancedIndicatorsStep(contract)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_df = step.transform(df)
            
        # Write to V3 directory
        out_dir = V3_PANEL_DIR / partition_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0000.parquet"
        transformed_df.to_parquet(out_path, index=False)
        
    print("Intraday V3 panel creation complete!")

if __name__ == "__main__":
    build_intraday_v3()
