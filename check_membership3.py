import pandas as pd
df = pd.read_parquet('data/canonical/index_membership/sp500_memberships_extended/memberships.parquet')
print('Max date:', df['effective_to_utc'].max())
print('Missing:', df['effective_to_utc'].isna().sum())
