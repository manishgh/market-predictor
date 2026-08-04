import json
from pathlib import Path

eval_path = Path(r"c:\project\market-predictor\models\edge_rebuild_swing_candidate_20260802_v2\evaluation.json")

with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

first = eval_data["validation_candidates"][0]
temp_gen = first["selected_validation_metrics"]["temporal_generalization_full_pit_cross_section"]
print(f"Metrics: {list(temp_gen.keys())}")
if "economic" in temp_gen:
    print(f"Economic keys: {list(temp_gen['economic'].keys())}")
if "diagnostics" in temp_gen:
    print(f"Diagnostics keys: {list(temp_gen['diagnostics'].keys())}")
