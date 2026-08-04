import pathlib
import re

swing_path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
wf_path = pathlib.Path('src/market_predictor/edge_rebuild/training/walk_forward.py')
utils_path = pathlib.Path('src/market_predictor/edge_rebuild/training/utils.py')

swing_content = swing_path.read_text('utf-8')
wf_content = wf_path.read_text('utf-8')
utils_content = utils_path.read_text('utf-8')

hor_match = re.search(r'^HORIZON_SESSIONS\s*:\s*Final\s*=\s*\d+\n', swing_content, re.MULTILINE)
wf_fold_match = re.search(r'@dataclass\nclass WalkForwardFold:.+?val_end_session:\s*str\n', swing_content, re.DOTALL)
ord_match = re.search(r'def _ordered_sessions\(.+?return tuple\(sessions\)\n', swing_content, re.DOTALL)

if hor_match and wf_fold_match and ord_match:
    h = hor_match.group(0)
    w = wf_fold_match.group(0)
    o = ord_match.group(0)
    
    swing_content = swing_content.replace(h, '')
    swing_content = swing_content.replace(w, '')
    swing_content = swing_content.replace(o, '')
    
    swing_content = swing_content.replace(
        'from market_predictor.edge_rebuild.training.utils import (\n    _finite,\n    _iso,\n    _mapping,\n    _security_holdout_mask,\n)',
        'from market_predictor.edge_rebuild.training.utils import (\n    HORIZON_SESSIONS,\n    WalkForwardFold,\n    _finite,\n    _iso,\n    _mapping,\n    _ordered_sessions,\n    _security_holdout_mask,\n)'
    )
    swing_path.write_text(swing_content, 'utf-8')
    
    utils_content = utils_content.replace('from typing import Any, cast, Final', 'from typing import Any, cast, Final\nfrom dataclasses import dataclass')
    utils_content += '\n\n' + h + '\n\n' + w + '\n\n' + o
    utils_path.write_text(utils_content, 'utf-8')
    
    wf_content = wf_content.replace(
        'from market_predictor.edge_rebuild.swing_training import (\n    HORIZON_SESSIONS,\n    SwingTrainingConfig,\n    WalkForwardFold,\n    _ordered_sessions,\n)',
        'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig\n\nfrom market_predictor.edge_rebuild.training.utils import (\n    HORIZON_SESSIONS,\n    WalkForwardFold,\n    _ordered_sessions,\n)'
    )
    wf_path.write_text(wf_content, 'utf-8')
    print("Success")
else:
    print("Failed")
