import json
from pathlib import Path

eval_path = Path(r"c:\project\market-predictor\models\edge_rebuild_swing_candidate_20260802_v2\evaluation.json")

with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

report = ["# Failure Attribution Analysis from evaluation.json\n"]

candidates = eval_data.get("validation_candidates", [])
report.append(f"Found {len(candidates)} candidates evaluated.\n")

for cand in candidates:
    cid = cand["candidate_id"]
    profile = cand["ablation_profile"]
    report.append(f"## Candidate: {cid} (Profile: {profile})")
    
    metrics = cand.get("selected_validation_metrics", {})
    if not metrics:
        report.append("No selected validation metrics found.\n")
        continue
    
    # 1. Temporal generalization
    temp = metrics.get("temporal_generalization_full_pit_cross_section", {})
    roc_auc = temp.get("roc_auc", "N/A")
    avg_net_return = temp.get("selected_average_managed_net_return", "N/A")
    report.append(f"- **Temporal AUC:** {roc_auc}")
    report.append(f"- **Temporal Avg Net Return:** {avg_net_return}")
    
    # Check if Rank IC exists
    rank_ic = temp.get("rank_ic", "Not Computed in V2")
    top_quant = temp.get("top_quantile_lift", "Not Computed in V2")
    report.append(f"- **Rank IC:** {rank_ic}")
    report.append(f"- **Top-Quantile Lift:** {top_quant}")
    
    # By Sector
    by_sector = temp.get("by_sector", [])
    if by_sector:
        report.append("\n  **By Sector (Selected Avg Net Return):**")
        for item in by_sector[:5]:
            # item is probably a dict with a key for sector name or something similar
            key = item.get("sector", str(item))
            report.append(f"  - {key}: {item.get('selected_average_managed_net_return', 'N/A')}")
            
    # By Regime
    by_regime = temp.get("by_regime", [])
    if by_regime:
        report.append("\n  **By Regime (Selected Avg Net Return):**")
        for item in by_regime[:5]:
            key = item.get("regime", str(item))
            report.append(f"  - {key}: {item.get('selected_average_managed_net_return', 'N/A')}")
            
    # 2. Unseen Security
    unseen = metrics.get("unseen_security_generalization_stable_20pct", {})
    roc_auc_unseen = unseen.get("roc_auc", "N/A")
    avg_net_return_unseen = unseen.get("selected_average_managed_net_return", "N/A")
    report.append(f"\n- **Unseen Security AUC:** {roc_auc_unseen}")
    report.append(f"- **Unseen Security Avg Net Return:** {avg_net_return_unseen}\n")

with open(r"c:\project\market-predictor\scripts\attribution_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("Done writing report to scripts/attribution_report.md")
