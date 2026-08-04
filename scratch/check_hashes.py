import json
import hashlib
from pathlib import Path

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

f = Path('data/features/edge_rebuild_swing_panel_20190709_20260708_v12')
a = json.load(open(f / 'final/_authority.json'))

print('Manifest expected:', a['feature_manifest_sha256'])
print('Manifest actual:', file_sha256(f / 'final/_manifest.json'))

source = a['sources'][0]
print('Daily expected:', source.get('daily_authority_sha256'))
print('Daily actual:', file_sha256(f / 'daily/_authority.json') if (f / 'daily/_authority.json').exists() else 'Missing')

print('Combined expected:', source.get('combined_daily_authority_sha256'))
print('Combined actual:', file_sha256(f / 'combined_daily/_authority.json') if (f / 'combined_daily/_authority.json').exists() else 'Missing')
