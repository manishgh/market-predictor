import json
import hashlib
from pathlib import Path
import pandas as pd

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def _json_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def fix_manifest():
    final_dir = Path("data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final")
    manifest_path = final_dir / "_manifest.json"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    files_by_profile = {"technical_market": [], "catalyst_full": []}
    
    panel_dir = final_dir / "panel"
    for profile_dir in panel_dir.iterdir():
        if not profile_dir.is_dir() or not profile_dir.name.startswith("feature_profile="):
            continue
            
        profile_name = profile_dir.name.replace("feature_profile=", "")
        
        for pqt_file in profile_dir.rglob("*.parquet"):
            rel_path = pqt_file.relative_to(final_dir).as_posix()
            df = pd.read_parquet(pqt_file, columns=["security_id", "session_date_et", "decision_id"])
            
            # Extract month from the parent directory name, which should be 'month=YYYY-MM'
            month_match = pqt_file.parent.name.replace("month=", "") if pqt_file.parent.name.startswith("month=") else ""
            
            decision_ids = df["decision_id"].astype(str)
            decision_ids_list = sorted(decision_ids.tolist())
            
            file_meta = {
                "path": rel_path,
                "sha256": file_sha256(pqt_file),
                "feature_profile": profile_name,
                "partition_month": month_match,
                "rows": len(df),
                "securities": len(df["security_id"].unique()),
                "sessions": len(df["session_date_et"].unique()),
                "first_session": str(df["session_date_et"].min().date() if hasattr(df["session_date_et"].min(), 'date') else df["session_date_et"].min()),
                "last_session": str(df["session_date_et"].max().date() if hasattr(df["session_date_et"].max(), 'date') else df["session_date_et"].max()),
                "decision_ids_sha256": _json_sha256(decision_ids_list)
            }
            files_by_profile[profile_name].append(file_meta)
            print(f"Added {rel_path} to manifest")
            
    manifest["files_by_profile"] = files_by_profile
    manifest["files"] = files_by_profile["technical_market"]
            
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    # We also need to update the _authority.json's feature_manifest_sha256 and artifact_sha256
    authority_path = final_dir / "_authority.json"
    with open(authority_path, 'r') as f:
        auth = json.load(f)
        
    new_manifest_sha = file_sha256(manifest_path)
    auth["feature_manifest_sha256"] = new_manifest_sha
    auth["artifact_sha256"] = new_manifest_sha
    print(f"Updated _authority.json feature_manifest_sha256 to {new_manifest_sha}")
    
    with open(authority_path, 'w') as f:
        json.dump(auth, f, indent=2)

if __name__ == "__main__":
    fix_manifest()
