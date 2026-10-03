"""Prove two pinned policies differ only in corporate-action source bindings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy

_ACTION_BINDINGS = frozenset({"action_config", "action_archive", "action_audit_sha256"})


@dataclass(frozen=True, slots=True)
class DecisionPriceEquivalence:
    decision_config: SourcePin
    target_config: SourcePin
    decision_policy: CorrectedOutcomePolicy
    target_policy: CorrectedOutcomePolicy
    shared_semantics_sha256: str

    def as_record(self) -> dict[str, Any]:
        """Bind both independent file identities and their shared typed semantics."""
        return {"schema": "market_predictor.decision_price_equivalence",
            "decision_config": self.decision_config.model_dump(mode="json"),
            "target_config": self.target_config.model_dump(mode="json"),
            "shared_semantics_sha256": self.shared_semantics_sha256}


def verify_decision_price_equivalence(root: Path, decision_config: SourcePin,
    target_config: SourcePin,
) -> DecisionPriceEquivalence:
    """Read only the two configurations; this does not admit sources or targets.

    Every typed policy field participates except the three named action bindings.
    New policy fields therefore join the proof automatically rather than silently
    escaping an allowlist of selected price or decision fields.
    """
    root = root.resolve()
    decision = load_corrected_outcome_policy(root, root / decision_config.path, decision_config.sha256)
    target = load_corrected_outcome_policy(root, root / target_config.path, target_config.sha256)
    left = decision.model_dump(mode="json", exclude=set(_ACTION_BINDINGS))
    right = target.model_dump(mode="json", exclude=set(_ACTION_BINDINGS))
    if left != right:
        changed = sorted(name for name in left.keys() | right.keys() if left.get(name) != right.get(name))
        raise DataReadinessError("target policy changes decision/price semantics: " + ", ".join(changed))
    return DecisionPriceEquivalence(decision_config, target_config, decision, target, json_sha256(left))
