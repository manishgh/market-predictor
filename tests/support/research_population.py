"""A valid approved-population audit for the real research-cohort loader."""
from __future__ import annotations

import json
from pathlib import Path

from market_predictor.canonical.store import file_sha256
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.research_cohort import ResearchSecurityExclusion, SwingResearchCohort


def write_population(root: Path, retained: tuple[str, ...], *, excluded: str = "cik:0000009999") -> Path:
    """Write `data/reports/population.json` retaining `retained` and excluding one other security."""
    source = root / "data/reports/population_source.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("test-only cohort source", encoding="utf-8")
    model = SwingResearchCohort(schema_version="market_predictor.swing_research_cohort",
        scope="retrospective_development_restriction", price_basis_status="not_certified_by_cohort",
        combined_daily_inputs_sha256="e" * 64, original_security_ids=tuple(sorted((*retained, excluded))),
        inherited_excluded_security_ids=(), warmup_only_security_ids=(),
        exclusions=(ResearchSecurityExclusion(security_id=excluded, tickers=("XXX",), reason="unavailable_trading"),),
        maximum_exclusion_bps=5000, cap_approval_reference="test-only",
        source_files={"data/reports/population_source.txt": file_sha256(source)})
    payload = {"cohort": model.model_dump(mode="json"), "cohort_sha256": model.sha256(), "summary": model.summary(),
               "coverage": {"test_only": True}}
    path = root / "data/reports/population.json"
    path.write_text(json.dumps({**payload, "audit_sha256": json_sha256(payload)}), encoding="utf-8")
    return path
