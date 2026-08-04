import pandas as pd
df = pd.read_parquet('data/canonical/index_membership/sp500_memberships_20180529_20260708_v1/memberships.parquet')
print(df['effective_to_utc'].max())
print(df[df['effective_to_utc'] == df['effective_to_utc'].max()].head())
