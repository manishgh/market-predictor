"""Synthetic authorities exercise publication; no retained source bodies are read."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import write_json_object
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_publication import ARTIFACT_TYPE, PROFILE, IssuerReactionPublicationPolicy
from market_predictor.swing.datasets import issuer_reaction_publication as owner
from market_predictor.swing.datasets import return_relationship_verification as relationship_verifier
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from tests.test_issuer_content_qualification_authority import _publish as publish_qualification
from tests.test_issuer_content_qualification_authority import fixture as _qualification_fixture
from tests.test_swing_issuer_reaction_profile import bundle as bundle
from tests.test_swing_return_relationship_publication import publication_fixture as publication_fixture

qualification_fixture = _qualification_fixture

def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


@pytest.fixture
def reaction_publication(publication_fixture: dict[str, Any], qualification_fixture: dict[str, Any],
                         monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Actual publication/review kernels; parent fixture admits only synthetic bars.

    The tiny 354-row fixture explicitly replaces the production row-count constant;
    all 59 months, original parent targets and real source/hash readers remain.
    """
    parent = publication_fixture
    root = parent["root"]
    assert root == qualification_fixture["root"]
    monkeypatch.setattr(owner, "__file__", str(root / "src/market_predictor/swing/datasets/issuer_reaction_publication.py"))
    monkeypatch.setattr(owner, "guard", lambda limit: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(owner, "EXPECTED_ROWS", parent["result"]["rows"])
    monkeypatch.setattr(relationship_verifier, "__file__",
                        str(root / "src/market_predictor/swing/datasets/return_relationship_verification.py"))
    monkeypatch.setattr(relationship_verifier, "_guard", lambda: None)
    monkeypatch.setattr(relationship_verifier, "release_process_memory", lambda: None)
    receipt_path = root / "data/reports/relationship_row_verification.json"
    relationship_verifier.verify_return_relationship_rows(root, parent["publication"], receipt_path)
    publish_qualification(qualification_fixture)
    qualification_folder = qualification_fixture["output"]
    policy = IssuerReactionPublicationPolicy(schema_version="market_predictor.issuer_reaction_publication_config",
        parent_publication=parent["publication"], parent_saved_row_verification=_pin(root, receipt_path),
        qualification_publication=_pin(root, qualification_folder / "_manifest.json"),
        qualification_authority=_pin(root, qualification_folder / "_authority.json"))
    config_path = root / "configs/issuer_reactions.json"
    write_json_object(config_path, policy.model_dump(mode="json"))
    output = root / "data/features/issuer_reactions"
    partial = owner.materialize_issuer_reactions(root, config_path, file_sha256(config_path), output, maximum_months_this_run=1)
    assert partial["status"] == "in_progress" and not (output / "_manifest.json").exists()
    saved_month = next(iter(partial["months"].values()))["profiles"][PROFILE]
    original = (output / saved_month["path"]).read_bytes()
    result = owner.materialize_issuer_reactions(root, config_path, file_sha256(config_path), output,
        expected_checkpoint_sha256=partial["checkpoint_sha256"])
    assert (output / saved_month["path"]).read_bytes() == original
    publication = _pin(root, output / "_manifest.json")
    return {"root": root, "publication": publication, "output": output, "sessions": parent["sessions"],
            "policy": policy, "config": _pin(root, config_path), "parent": parent["parent"],
            "relationship": parent, "relationship_parent": parent, "result": result, "partial": partial}


def test_native_publication_preserves_all_parent_rows_and_unknown_coverage(reaction_publication: dict[str, Any]) -> None:
    value = reaction_publication
    verified = owner.verify_issuer_reaction_publication(value["root"], value["publication"])
    assert len(verified.model_columns) == 126 and tuple(verified.model_columns[-2:]) == REACTION_COLUMNS
    assert len(verified.months) == 59 and verified.manifest["rows"] == 354
    assert verified.request["coverage"] == "unknown" and verified.request["baseline_numerical_replayed"] is False
    assert verified.request["qualification_source_replayed"] is True
    for month, record in verified.months.items():
        child = record["profiles"][PROFILE]
        frame, _ = load_canonical_artifact(value["output"] / child["path"], expected_type=ARTIFACT_TYPE, allow_research=True)
        parent_record = verified.parent_manifest["months"][month]["profiles"]["technical_relationships"]
        parent, _ = load_canonical_artifact(verified.parent_path.parent / parent_record["path"], allow_research=True)
        owner.assert_parent_parity(parent, frame)
        assert frame.reaction_coverage_status.eq("unknown").all()
        assert frame.reaction_coverage_available_at_utc.isna().all()
    assert all(verified.manifest[name] is False for name in owner.CLOSED)
    with pytest.raises(DataReadinessError, match="external checkpoint"):
        owner.materialize_issuer_reactions(value["root"], Path(value["config"].path), value["config"].sha256, value["output"])


def test_qualification_disposition_byte_change_prevents_reuse(reaction_publication: dict[str, Any]) -> None:
    value = reaction_publication
    path = value["root"] / Path(value["policy"].qualification_publication.path).parent / "version_dispositions.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataReadinessError):
        owner.verify_issuer_reaction_publication(value["root"], value["publication"])


@pytest.mark.parametrize("poison", ["value", "clock", "eligibility", "target", "identity", "dtype"])
def test_month_assembly_refuses_changed_inherited_columns(bundle: dict[str, Any], poison: str) -> None:
    parent = bundle["baseline"]
    built = build_issuer_reaction_profile(**bundle)
    rows = built.rows.copy()
    if poison == "value":
        rows.loc[0, parent.model_columns[0]] += 1
    elif poison == "clock":
        rows.loc[0, "parent_clock"] -= pd.Timedelta(seconds=1)
    elif poison == "eligibility":
        rows.loc[0, "feature_eligible"] = True
    elif poison == "target":
        rows.loc[0, "spy_fixed_horizon_excess_return"] = 0.0
    elif poison == "identity":
        rows.loc[0, "decision_id"] = "foreign"
    else:
        rows[parent.model_columns[0]] = rows[parent.model_columns[0]].astype("float64")
    with pytest.raises(DataReadinessError):
        owner.assemble_reaction_month(parent.rows, [replace(built, rows=rows)])


def test_complete_state_cannot_drop_a_month_or_rewrite_row_count(reaction_publication: dict[str, Any]) -> None:
    value = reaction_publication
    path = value["root"] / value["publication"].path
    manifest = json.loads(path.read_text())
    manifest["months"].pop(next(iter(manifest["months"])))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="total differs|incomplete"):
        owner.verify_issuer_reaction_publication(value["root"], _pin(value["root"], path))
