import json
import hashlib
from pathlib import Path

def file_sha256(p): 
    h=hashlib.sha256()
    h.update(open(p,'rb').read())
    return h.hexdigest()
    
f=Path('data/features/edge_rebuild_swing_panel_20190709_20260708_v12')
m=json.load(open(f / 'final/_manifest.json'))
c=f / 'combined_daily/_authority.json'

print("Expected combined:", m['source']['combined_daily_authority_sha256'])
print("Actual combined:", file_sha256(c) if c.exists() else 'Missing')

d=f / 'daily/_authority.json'
print("Expected daily:", m['source']['daily_authority_sha256'])
print("Actual daily:", file_sha256(d) if d.exists() else 'Missing')

