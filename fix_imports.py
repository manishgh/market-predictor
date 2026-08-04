import pathlib
path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
lines = path.read_text('utf-8')
lines = lines.replace(
    'from market_predictor.edge_rebuild.training.economics import (\n    _economic_gate,\n)',
    'from market_predictor.edge_rebuild.training.economics import (\n    _daily_position_ledger,\n    _economic_gate,\n    _moving_block_bootstrap_mean_interval,\n    _session_bootstrap,\n    _session_economic_blocks,\n    _stability_breakdown,\n    _stability_summary,\n    _year_breakdown,\n)'
)
path.write_text(lines, 'utf-8')
