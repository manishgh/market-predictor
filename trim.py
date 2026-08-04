import pathlib
path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
lines = path.read_text('utf-8').splitlines()
lines = [line.rstrip() for line in lines]
path.write_text('\n'.join(lines) + '\n', 'utf-8')
