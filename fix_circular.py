import pathlib

for fname in ['economics.py', 'evaluation.py', 'walk_forward.py']:
    path = pathlib.Path(f'src/market_predictor/edge_rebuild/training/{fname}')
    content = path.read_text('utf-8')
    content = content.replace(
        'from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig\n',
        'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from market_predictor.edge_rebuild.swing_training import SwingTrainingConfig\n'
    )
    path.write_text(content, 'utf-8')
