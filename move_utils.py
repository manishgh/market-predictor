import pathlib
import re

utils_path = pathlib.Path('src/market_predictor/edge_rebuild/training/utils.py')
swing_path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
eval_path = pathlib.Path('src/market_predictor/edge_rebuild/training/evaluation.py')
wf_path = pathlib.Path('src/market_predictor/edge_rebuild/training/walk_forward.py')

# 1. Read swing_training.py and extract the functions
swing_content = swing_path.read_text('utf-8')
iso_regex = re.compile(r'def _iso\(.+?return cast\(str, parsed\.tz_convert\("UTC"\)\.isoformat\(\)\)\n', re.DOTALL)
holdout_regex = re.compile(r'def _security_holdout_mask\(.+?return assigned\.astype\(bool\)\n', re.DOTALL)

iso_match = iso_regex.search(swing_content)
holdout_match = holdout_regex.search(swing_content)

if iso_match and holdout_match:
    iso_func = iso_match.group(0)
    holdout_func = holdout_match.group(0)
    
    # Remove from swing_training.py
    swing_content = swing_content.replace(iso_func, '')
    swing_content = swing_content.replace(holdout_func, '')
    
    # Add imports to swing_training.py
    swing_content = swing_content.replace(
        'from market_predictor.edge_rebuild.training.utils import (\n    _finite,\n    _mapping,\n)',
        'from market_predictor.edge_rebuild.training.utils import (\n    _finite,\n    _mapping,\n    _iso,\n    _security_holdout_mask,\n)'
    )
    swing_path.write_text(swing_content, 'utf-8')

    # Add to utils.py
    utils_content = utils_path.read_text('utf-8')
    utils_content = utils_content.replace(
        'import math\nfrom collections.abc import Mapping\nfrom typing import Any',
        'import math\nimport hashlib\nimport pandas as pd\nfrom collections.abc import Mapping\nfrom typing import Any, cast\nfrom market_predictor.edge_rebuild.strategy_contract import StrategyContract'
    )
    utils_content += '\n\n' + holdout_func + '\n\n' + iso_func
    utils_path.write_text(utils_content, 'utf-8')

    # Update evaluation.py to import from utils instead
    eval_content = eval_path.read_text('utf-8')
    eval_content = eval_content.replace(
        'from market_predictor.edge_rebuild.swing_training import _iso, _security_holdout_mask',
        'from market_predictor.edge_rebuild.training.utils import _iso, _security_holdout_mask'
    )
    eval_path.write_text(eval_content, 'utf-8')
    
    # Update walk_forward.py
    wf_content = wf_path.read_text('utf-8')
    wf_content = wf_content.replace(
        'from market_predictor.edge_rebuild.swing_training import _iso',
        'from market_predictor.edge_rebuild.training.utils import _iso'
    )
    wf_path.write_text(wf_content, 'utf-8')

    print("Success")
else:
    print("Could not find functions to extract.")

