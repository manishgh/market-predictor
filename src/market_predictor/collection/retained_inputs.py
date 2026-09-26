"""Inputs the retained collectors bind by path and hash; the intraday retirement never deletes or edits them.

The SIP-session request pins both policy files (file and model hashes). Every broker-action poll
re-derives the A4.3 dataset's metadata hashes and verifies the base membership authority it names.
"""
from __future__ import annotations

RETAINED_CONFIGS = (
    "configs/edge_rebuild_intraday_history.toml",
    "configs/edge_rebuild_selected_session_benchmarks.toml",
)
RETAINED_LINEAGE_DATA = (
    "data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1/_request.json",
    "data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1/_manifest.json",
    "data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1/_authority.json",
    "data/canonical/index_membership/sp500_memberships_20180529_20260708_v1",
)
