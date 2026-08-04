import json
import hashlib
from pathlib import Path
import pyarrow.parquet as pq

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    v3_dir = Path("data/features/edge_rebuild_intraday_dataset_causal_20260804_v3")
    
    # load v2 manifest from the ORIGINAL v2 dir to get the pristine list structure
    v2_manifest_path = Path("data/features/edge_rebuild_intraday_dataset_causal_20260802_v2/_manifest.json")
    manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    
    print("Regenerating hashes, bytes, and rows...")
    
    # Update files list
    for entry in manifest.get("files", []):
        file_path = v3_dir / entry["path"]
        if file_path.exists():
            entry["sha256"] = file_sha256(file_path)
            entry["bytes"] = file_path.stat().st_size
            if file_path.suffix == ".parquet":
                entry["rows"] = pq.read_metadata(file_path).num_rows
            
    # Update partitions list
    for entry in manifest.get("partitions", []):
        file_path = v3_dir / entry["path"]
        if file_path.exists():
            entry["sha256"] = file_sha256(file_path)
            entry["bytes"] = file_path.stat().st_size
            meta = pq.read_metadata(file_path)
            entry["rows"] = meta.num_rows
            # approximate eligible_rows since it's just metadata, or calculate it exactly
            import pandas as pd
            df = pd.read_parquet(file_path, columns=["dataset_eligible"])
            entry["eligible_rows"] = int(df["dataset_eligible"].sum())
            
    # Write to v3 manifest
    v3_manifest_path = v3_dir / "_manifest.json"
    v3_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    
    # Update authority
    auth_path = v3_dir / "_authority.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["artifact_sha256"] = file_sha256(v3_manifest_path)
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
    
    print("Manifest and authority updated!")
