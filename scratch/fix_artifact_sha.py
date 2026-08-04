import json
import hashlib
from pathlib import Path

def file_sha256(p): 
    h=hashlib.sha256()
    h.update(open(p,'rb').read())
    return h.hexdigest()
    
f=Path('data/features/edge_rebuild_swing_panel_20190709_20260708_v12/final')
m_sha = file_sha256(f / '_manifest.json')
a = json.load(open(f / '_authority.json'))
a['artifact_sha256'] = m_sha
json.dump(a, open(f / '_authority.json', 'w'), indent=2)
print("Updated artifact_sha256 to", m_sha)
