from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import market_predictor.intraday.features.bar_labels as intraday_bar_labels
import market_predictor.intraday.features.labels as intraday_labels
import market_predictor.modeling.label_outcomes as label_outcomes
import market_predictor.swing.labels.barrier_and_rank as swing_barrier_labels

OUTCOME_NAMES = (
    "TARGET_HIT",
    "STOP_HIT",
    "TIMEOUT",
    "RANK_TOP",
    "RANK_BOTTOM",
    "RANK_MIDDLE",
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "market_predictor"


def test_label_outcome_values_are_frozen() -> None:
    values = {name: getattr(label_outcomes, name) for name in OUTCOME_NAMES}
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))

    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "b021c7ad67fedfe5ca3685189f184488520994bb86c0e68535beb76c52d36c19"
    )


def test_label_outcomes_has_the_only_top_level_definitions() -> None:
    owners: dict[str, list[Path]] = {name: [] for name in OUTCOME_NAMES}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in owners:
                        owners[target.id].append(path)

    expected = PACKAGE_ROOT / "modeling" / "label_outcomes.py"
    assert owners == {name: [expected] for name in OUTCOME_NAMES}


def test_consumers_do_not_reexport_label_outcomes() -> None:
    consumers = (intraday_bar_labels, intraday_labels, swing_barrier_labels)
    assert not any(
        hasattr(module, outcome_name)
        for module in consumers
        for outcome_name in OUTCOME_NAMES
    )
