import pandas as pd
import numpy as np
df = pd.read_parquet('data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet')
print('Unique effective_to_utc:', df['effective_to_utc'].dropna().unique()[-5:])
print('Null count:', df['effective_to_utc'].isna().sum())
print('Empty string count:', (df['effective_to_utc'] == '').sum())
