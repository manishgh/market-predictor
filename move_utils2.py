import pathlib
import re

swing_path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
wf_path = pathlib.Path('src/market_predictor/edge_rebuild/training/walk_forward.py')
utils_path = pathlib.Path('src/market_predictor/edge_rebuild/training/utils.py')

swing_content = swing_path.read_text('utf-8')
wf_content = wf_path.read_text('utf-8')
utils_content = utils_path.read_text('utf-8')

# Move HORIZON_SESSIONS, _required_finite_number, _object to utils.py
# Wait, HORIZON_SESSIONS is a constant. Let's see if it's there.
hor_match = re.search(r'^HORIZON_SESSIONS\s*:\s*Final\s*=\s*\d+', swing_content, re.MULTILINE)
req_finite = re.search(r'def _required_finite_number\(.+?return float\(value\)\n', swing_content, re.DOTALL)
obj_match = re.search(r'def _object\(.+?return value\n', swing_content, re.DOTALL)

# Let's just use TYPE_CHECKING in walk_forward for SwingTrainingConfig and import HORIZON_SESSIONS properly.
# Actually, we can move _required_finite_number, _object, HORIZON_SESSIONS to utils.
if hor_match and req_finite and obj_match:
    hor_val = hor_match.group(0)
    req_val = req_finite.group(0)
    obj_val = obj_match.group(0)
    
    swing_content = swing_content.replace(hor_val, '')
    swing_content = swing_content.replace(req_val, '')
    swing_content = swing_content.replace(obj_val, '')
    
    swing_content = swing_content.replace(
        'from market_predictor.edge_rebuild.training.utils import (\n    _finite,\n    _iso,\n    _mapping,\n    _security_holdout_mask,\n)',
        'from market_predictor.edge_rebuild.training.utils import (\n    HORIZON_SESSIONS,\n    _finite,\n    _iso,\n    _mapping,\n    _object,\n    _required_finite_number,\n    _security_holdout_mask,\n)'
    )
    swing_path.write_text(swing_content, 'utf-8')
    
    utils_content = utils_content.replace('from typing import Any, cast', 'from typing import Any, cast, Final')
    utils_content += '\n\n' + hor_val + '\n\n' + req_val + '\n\n' + obj_val
    utils_path.write_text(utils_content, 'utf-8')
    
    wf_content = wf_content.replace(
        'from market_predictor.edge_rebuild.swing_training import (\n    HORIZON_SESSIONS,\n    _object,\n    _required_finite_number,\n)',
        'from market_predictor.edge_rebuild.training.utils import (\n    HORIZON_SESSIONS,\n    _object,\n    _required_finite_number,\n)'
    )
    wf_path.write_text(wf_content, 'utf-8')
    print("Success moving functions")
else:
    print("Failed to match functions")
