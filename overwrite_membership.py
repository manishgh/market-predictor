import shutil
from pathlib import Path

original = Path('data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet')
extended = Path('data/canonical/index_membership/sp500_memberships_extended/memberships.parquet')

# backup original
if not Path('data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet.bak').exists():
    shutil.copy2(original, original.with_suffix('.parquet.bak'))

# overwrite with extended
shutil.copy2(extended, original)
print('Overwritten successfully')
