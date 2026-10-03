import json
import tomllib
from pathlib import Path

from market_predictor.canonical.store import file_sha256
from market_predictor.swing.datasets.corporate_action_scope import CorporateActionScope

ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_scope_keeps_transport_parameters_and_current_price_selection():
    old = tomllib.loads((ROOT / "configs/swing_full_cohort_corporate_actions.toml").read_text())
    current = tomllib.loads((ROOT / "configs/swing_corporate_action_sources.toml").read_text())
    price_path = ROOT / "configs/swing_corrected_outcomes.toml"
    price = tomllib.loads(price_path.read_text())
    features = tomllib.loads((ROOT / "configs/swing_corrected_research_features.toml").read_text())
    CorporateActionScope.model_validate_json(json.dumps(current))
    assert current["source_selection"] == price["source_selection"]
    assert {key: value for key, value in current.items() if key != "source_selection"} == {
        key: value for key, value in old.items() if key != "source_selection"}
    assert features["outcome_source_config"]["sha256"] == file_sha256(price_path)
    assert price["action_config"]["path"] == "configs/swing_full_cohort_corporate_actions.toml"
