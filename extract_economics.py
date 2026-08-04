import pathlib
path = pathlib.Path('src/market_predictor/edge_rebuild/swing_training.py')
lines = path.read_text('utf-8').splitlines()

# find _session_economic_blocks
start_idx = -1
for i, line in enumerate(lines):
    if line.startswith('def _session_economic_blocks('):
        start_idx = i
        break

# find _calibration_bins
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith('def _calibration_bins('):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + lines[end_idx:]
    print(f'Removed {end_idx - start_idx} lines of economics logic.')

# now remove _mapping and _finite
# _mapping
start_map = -1
for i, line in enumerate(lines):
    if line.startswith('def _mapping('):
        start_map = i
        break
if start_map != -1:
    end_map = start_map
    while lines[end_map].strip() or lines[end_map-1].strip():
        end_map += 1
    lines = lines[:start_map] + lines[end_map:]

# _finite
start_fin = -1
for i, line in enumerate(lines):
    if line.startswith('def _finite('):
        start_fin = i
        break
if start_fin != -1:
    end_fin = start_fin
    while lines[end_fin].strip() or lines[end_fin-1].strip():
        end_fin += 1
    lines = lines[:start_fin] + lines[end_fin:]

# Add imports for economics
import_lines = [
    'from market_predictor.edge_rebuild.training.economics import (',
    '    _economic_gate,',
    ')',
    'from market_predictor.edge_rebuild.training.utils import (',
    '    _finite,',
    '    _mapping,',
    ')'
]

# Find the last import from market_predictor.edge_rebuild
insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith('from market_predictor.v3.errors'):
        insert_idx = i
        break

lines = lines[:insert_idx] + import_lines + lines[insert_idx:]

path.write_text('\n'.join(lines) + '\n', 'utf-8')
print('Extraction complete.')
