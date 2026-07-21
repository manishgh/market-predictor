"""Canonical point-in-time market data contracts and transformations."""

from market_predictor.canonical.contracts import (
    CANONICAL_SCHEMA_VERSION,
    AvailabilityPolicy,
    CanonicalBar,
    CanonicalEvent,
    CanonicalFundamentalFact,
    CanonicalUniverseMembership,
    SourceCollection,
    SourceCollectionStatus,
)
from market_predictor.canonical.normalize import canonicalize_bars, canonicalize_events, canonicalize_universe_memberships
from market_predictor.canonical.store import load_canonical_artifact, write_canonical_artifact

__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "AvailabilityPolicy",
    "CanonicalBar",
    "CanonicalEvent",
    "CanonicalFundamentalFact",
    "CanonicalUniverseMembership",
    "SourceCollection",
    "SourceCollectionStatus",
    "canonicalize_bars",
    "canonicalize_events",
    "canonicalize_universe_memberships",
    "load_canonical_artifact",
    "write_canonical_artifact",
]
