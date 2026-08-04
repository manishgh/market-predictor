import pandas as pd
from pathlib import Path
import shutil
import warnings

def repartition():
    panel_dir = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final/panel")
    
    for profile in ["catalyst_full", "technical_market"]:
        profile_dir = panel_dir / f"feature_profile={profile}"
        if not profile_dir.exists():
            continue
            
        print(f"Processing {profile}...")
        
        # Load the single unpartitioned parquet
        pqt_file = profile_dir / "part-0000.parquet"
        if pqt_file.exists():
            df = pd.read_parquet(pqt_file)
            print(f"  Loaded {len(df)} rows")
            
            # Create partition_month column
            if "session_date_et" in df.columns:
                df["partition_month"] = pd.to_datetime(df["session_date_et"]).dt.strftime("%Y-%m")
                
                # Delete the original unpartitioned file
                pqt_file.unlink()
                
                # Write partitions
                for month, group in df.groupby("partition_month"):
                    month_dir = profile_dir / f"month={month}"
                    month_dir.mkdir(exist_ok=True)
                    # Exclude the partition_month column when saving
                    group_to_save = group.drop(columns=["partition_month"])
                    group_to_save.to_parquet(month_dir / "part.parquet", index=False)
                    print(f"  Wrote month={month}")
                    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repartition()
