import pathlib

path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
lines = path.read_text('utf-8').splitlines()

def extract_function(name):
    start = -1
    for i, line in enumerate(lines):
        if line.startswith(f'def {name}('):
            start = i
            break
    if start == -1: return []
    end = start + 1
    while end < len(lines) and (lines[end].startswith(' ') or lines[end].startswith(')') or lines[end].strip() == ''):
        end += 1
    while end < len(lines) and not lines[end].strip():
        end += 1
    fn = lines[start:end]
    del lines[start:end]
    return fn

fn_names = [
    '_calibration_bins',
    '_expected_calibration_error',
    '_overlap_audit',
    '_overlap_record'
]

extracted = []
for fn in fn_names:
    extracted.extend(extract_function(fn))
    extracted.append('')

eval_path = pathlib.Path('src/market_predictor/edge_rebuild/training/evaluation.py')
imports = '''from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract

'''
eval_path.write_text(imports + '\n'.join(extracted) + '\n', 'utf-8')

import_lines = [
    'from market_predictor.edge_rebuild.training.evaluation import (',
    '    _calibration_bins,',
    '    _expected_calibration_error,',
    '    _overlap_audit,',
    ')'
]

insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith('from market_predictor.v3.errors'):
        insert_idx = i
        break

lines = lines[:insert_idx] + import_lines + lines[insert_idx:]

path.write_text('\n'.join(lines) + '\n', 'utf-8')
print('Evaluation extraction complete.')
