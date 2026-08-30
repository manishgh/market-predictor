from __future__ import annotations

import importlib.util
from pathlib import Path

from market_predictor.swing.labels import add_exact_swing_labels


def test_swing_label_package_has_one_unambiguous_origin() -> None:
    spec = importlib.util.find_spec("market_predictor.swing.labels")

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).as_posix().endswith(
        "market_predictor/swing/labels/__init__.py"
    )
    assert spec.submodule_search_locations is not None


def test_existing_exact_swing_label_owner_is_unchanged() -> None:
    assert add_exact_swing_labels.__module__ == "market_predictor.swing.labels"
