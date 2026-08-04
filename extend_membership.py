import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from pathlib import Path

src_dir = Path('data/canonical/index_membership/sp500_memberships_20180529_20260708_v1')
dst_dir = Path('data/canonical/index_membership/sp500_memberships_extended')
if dst_dir.exists():
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

df = pd.read_parquet(dst_dir / 'memberships.parquet')
print('Old max date:', df['effective_to_utc'].max())
df['effective_to_utc'] = df['effective_to_utc'].astype(str).replace('2026-07-08 20:00:00+00:00', '2026-08-31 20:00:00+00:00')
# Wait, some might just be 2026-07-08 without time? Let's be safer.
df.loc[df['effective_to_utc'].str.contains('2026-07-08'), 'effective_to_utc'] = '2026-08-31 20:00:00+00:00'
print('New max date:', df['effective_to_utc'].max())
df.to_parquet(dst_dir / 'memberships.parquet', index=False)
