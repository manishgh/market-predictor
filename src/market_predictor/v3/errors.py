from __future__ import annotations


class MarketPredictorError(Exception):
    """Base exception for expected domain failures."""


class DataReadinessError(MarketPredictorError):
    """Input data is incomplete, stale, or unsuitable for the requested operation."""


class SchemaMismatchError(MarketPredictorError):
    """A dataset, feature, model, or API schema is incompatible."""


class LeakageAuditError(MarketPredictorError):
    """Point-in-time or label-availability validation failed."""


class ArtifactIntegrityError(MarketPredictorError):
    """An artifact is missing or does not match its immutable identity."""


class PromotionGateError(MarketPredictorError):
    """A model failed one or more predeclared promotion gates."""
