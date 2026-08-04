import pathlib
import re

eco_path = pathlib.Path('src/market_predictor/edge_rebuild/training/economics.py')
eco = eco_path.read_text('utf-8')
eco = eco.replace('from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig\n', 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig\n')
eco_path.write_text(eco, 'utf-8')

eval_path = pathlib.Path('src/market_predictor/edge_rebuild/training/evaluation.py')
ev = eval_path.read_text('utf-8')
ev = ev.replace('from market_predictor.edge_rebuild.swing_training import _iso, _security_holdout_mask\n', '')
ev = ev.replace('def _temporal_summary(\n', 'def _temporal_summary(\n    from market_predictor.edge_rebuild.swing_training import _iso\n')
ev = ev.replace('def evaluation_audit(\n', 'def evaluation_audit(\n    from market_predictor.edge_rebuild.swing_training import _security_holdout_mask\n')
eval_path.write_text(ev, 'utf-8')

wf_path = pathlib.Path('src/market_predictor/edge_rebuild/training/walk_forward.py')
wf = wf_path.read_text('utf-8')
wf = wf.replace(
    'from market_predictor.edge_rebuild.swing_training import (\n    HORIZON_SESSIONS,\n    SwingTrainingConfig,\n    WalkForwardFold,\n    _ordered_sessions,\n)\n',
    'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig, WalkForwardFold\n'
)
wf = wf.replace('def _validate_stride_and_horizon', 'def _validate_stride_and_horizon')
wf = wf.replace('def generate_walk_forward_folds(\n', 'def generate_walk_forward_folds(\n    from market_predictor.edge_rebuild.swing_training import HORIZON_SESSIONS, WalkForwardFold, _ordered_sessions\n')
wf = wf.replace('def apply_calendar_stride(\n', 'def apply_calendar_stride(\n    from market_predictor.edge_rebuild.swing_training import HORIZON_SESSIONS\n')
wf_path.write_text(wf, 'utf-8')

