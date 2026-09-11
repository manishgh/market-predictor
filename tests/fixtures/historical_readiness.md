# Historical Readiness Fixture

`historical_readiness_authority/` is an exact-byte copy of the retained
`data/research/edge_rebuild_readiness_er1_20260728` audit (13 files, 233,806 bytes).
It contains readiness metadata and a session calendar, not stock prices, news,
credentials, or a trained model. Original provider archives remain untouched.

The fixture makes historical replay and rejection tests independent of ignored
laptop data. Its existing request, manifest and authority hashes remain unchanged.
Tests must prove both strict historical replay and rejection as authority for a
new collection plan. This historical format cannot authorize current planning or
production. Never refresh the fixture's hashes merely to make a failing test pass.
