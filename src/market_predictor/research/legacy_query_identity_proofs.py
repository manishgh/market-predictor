"""Leased, immutable publication of identity proofs for legacy news-query IDs."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.issuer_content_inventory import _guard
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.universe.legacy_query_identity import (
    AVAILABILITY_BASIS,
    EVIDENCE_COMPLETE_DATE,
    PROOF_KINDS,
    build_legacy_query_proofs,
)
from market_predictor.universe.sp500.index_change_events import require_spglobal_event_reconstruction_ready
from market_predictor.universe.sp500.membership_authority import load_sp500_membership_authority_envelope
from market_predictor.universe.sp500.membership_history import IndexChange
from market_predictor.universe.symbol_correction_policy import load_symbol_correction_policy

CONFIG_SCHEMA = "market_predictor.legacy_query_identity_proofs_config"
SCHEMA = "market_predictor.legacy_query_identity_proofs"
IMPLEMENTATION_PATHS = (
    "research/legacy_query_identity_proofs.py", "universe/legacy_query_identity.py", "universe/issuer_news_identity.py",
    "universe/sp500/membership_history.py", "universe/sp500/membership_authority.py",
    "universe/sp500/index_change_events.py", "sources/spglobal/archive.py", "universe/symbol_correction_policy.py",
    "canonical/store.py", "core/symbols.py", "evidence/hashing.py", "evidence/io.py",
)
KIND_EVIDENCE = {
    "company_ticker_hash_reproduced": "one pinned S&P event company and ticker mint both the legacy and the target ID",
    "sp500_spell_events_reproduced": "the addition event opening the spell mints the legacy ID, the deletion event closing "
                                     "it mints the target ID, and both spells have identical boundaries",
    "cik_equal": "the legacy ID embeds the target CIK and both carry the query ticker wherever they overlap",
    "cusip_chain_end_ticker_match": "weaker: pinned Alpaca transitions explain every ticker change of the chain, and one "
                                    "target security carries its last ticker at that security's final instant",
}
TRANSITION_EVIDENCE = ("byte-pinned retrospective Alpaca corporate-action snapshot saved beside the legacy S&P build; "
                       "no collection receipt binds its origin")
SYMBOL_VALIDITY = ("query tickers are the legacy spell tickers; CUSIP chains are checked against the pinned transitions, "
                   "the other kinds rely on the legacy spells")
_KEYS = frozenset({"schema", "identity_manifest", "target_membership_authority", "symbol_corrections", "event_authority",
                   "event_raw_archive", "security_transitions", "monthly_news_config", "archives"})
_NEW_YORK = "America/New_York"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def pin_file(root: Path, record: Any, pins: dict[str, str]) -> Path:
    """Verify one root-relative path/hash pin and record it."""
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "configuration requires path/hash pins")
    path = inside(root, record["path"])
    _require(file_sha256(path) == record["sha256"], f"input pin differs: {record['path']}")
    pins[path.relative_to(root).as_posix()] = record["sha256"]
    return path


def load_identity_bridge(root: Path, record: Any, pins: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the CIK bridge its pinned alignment manifest binds, with that manifest's source pins."""
    manifest = pinned_object(pin_file(root, record, pins), record["sha256"])
    sources = manifest.get("source_files")
    if manifest.get("schema") != "market_predictor.issuer_news_identity_alignment" or not isinstance(sources, dict):
        raise DataReadinessError("identity alignment manifest differs")
    bridge = inside(root, Path(record["path"]).parent / "identity_bridge.parquet")
    for path in (bridge, manifest_path_for(bridge)):
        relative = path.relative_to(root).as_posix()
        _require(sources.get(relative) == file_sha256(path), "identity bridge is not pinned by its alignment manifest")
        pins[relative] = sources[relative]
    frame, _ = load_canonical_artifact(bridge, expected_type="issuer_news_identity_bridge", allow_research=True)
    return frame, sources


def load_legacy_query_proofs(root: Path, record: Any, pins: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load published proofs through their manifest pin."""
    manifest = pinned_object(pin_file(root, record, pins), record["sha256"])
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete", "legacy identity proofs differ")
    path = inside(root, Path(record["path"]).parent / "proofs.parquet")
    _require(file_sha256(path) == manifest.get("proofs_sha256"), "legacy identity proofs differ from their manifest")
    pins[path.relative_to(root).as_posix()] = manifest["proofs_sha256"]
    return pd.read_parquet(path), manifest


def _target(root: Path, record: Any, bridge_sources: dict[str, str],
            pins: dict[str, str]) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    path = pin_file(root, record, pins)
    _require(bridge_sources.get(path.relative_to(root).as_posix()) == record["sha256"],
             "target membership authority differs from the CIK bridge's")
    memberships, envelope = load_sp500_membership_authority_envelope(path.parent)
    _require(envelope["authority_sha256"] == record["sha256"], "target membership authority envelope differs")
    manifest = pinned_object(path.parent / "_manifest.json", str(envelope["manifest_sha256"]))
    lineage = manifest.get("parent_lineage")
    if not isinstance(lineage, dict):
        raise DataReadinessError("target membership authority lacks its event lineage")
    cutoff = pd.Timestamp(str(envelope["cutoff_date"]), tz=_NEW_YORK).tz_convert("UTC")
    return memberships, cutoff, lineage


def _changes(root: Path, settings: dict[str, Any], lineage: dict[str, Any], pins: dict[str, str]) -> tuple[IndexChange, ...]:
    event = pin_file(root, settings["event_authority"], pins)
    raw = pin_file(root, settings["event_raw_archive"], pins)
    _require(lineage.get("event_authority_sha256") == settings["event_authority"]["sha256"]
             and lineage.get("raw_authority_sha256") == settings["event_raw_archive"]["sha256"],
             "event authority is not the target membership authority's parent")
    verified = require_spglobal_event_reconstruction_ready(event.parent, archive_directory=raw.parent)
    _require(verified.authority_sha256 == settings["event_authority"]["sha256"]
             and verified.event_set_sha256 == lineage.get("event_set_sha256"), "verified event authority differs")
    return verified.changes


def _corrected(root: Path, record: Any, bridge_sources: dict[str, str], pins: dict[str, str]) -> set[str]:
    path = pin_file(root, record, pins)
    relative = path.relative_to(root).as_posix()
    _require(bridge_sources.get(relative) == record["sha256"], "symbol corrections differ from the CIK bridge's")
    policy = load_symbol_correction_policy(root, Path(relative), record["sha256"])
    return {correction.security_id for correction in policy.corrections}


def _legacy(root: Path, settings: dict[str, Any], pins: dict[str, str]) -> list[pd.DataFrame]:
    """Each derived archive's queries were minted from the membership file its pinned raw request names."""
    monthly = pinned_object(pin_file(root, settings["monthly_news_config"], pins), settings["monthly_news_config"]["sha256"])
    sources, archives = monthly.get("sources"), settings["archives"]
    if not isinstance(sources, dict) or not isinstance(archives, dict) or set(archives) != {
            name for name, spec in sources.items() if spec.get("kind") == "derived"}:
        raise DataReadinessError("every derived news archive requires its query membership pin")
    frames: dict[str, pd.DataFrame] = {}
    for name in sorted(archives):
        spec, source = archives[name], sources[name]
        _require(isinstance(spec, dict) and set(spec) == {"collection", "memberships"}, f"archive pin differs: {name}")
        derived = inside(root, source["directory"]) / "_manifest.json"
        request_sha = pinned_object(derived, source["manifest_sha256"])["request"]["collection_manifest_sha256"]
        pins[derived.relative_to(root).as_posix()] = source["manifest_sha256"]
        collection = inside(root, spec["collection"])
        raw = pinned_object(collection / "_manifest.json", request_sha)
        pins[f"{collection.relative_to(root).as_posix()}/_manifest.json"] = request_sha
        request = pinned_object(collection / "_request.json")
        body = {key: value for key, value in request.items() if key != "request_sha256"}
        _require(request.get("request_sha256") == json_sha256(body) == raw.get("request_sha256"),
                 f"raw news request differs from its collection: {name}")
        pins[f"{collection.relative_to(root).as_posix()}/_request.json"] = file_sha256(collection / "_request.json")
        record = spec["memberships"]
        _require(isinstance(record, dict) and set(record) == {"path", "sha256", "manifest_sha256"}
                 and request.get("memberships_sha256") == record["sha256"],
                 f"archive queries were not minted from the pinned membership file: {name}")
        path = pin_file(root, {"path": record["path"], "sha256": record["sha256"]}, pins)
        sidecar = manifest_path_for(path)
        _require(file_sha256(sidecar) == record["manifest_sha256"], f"membership manifest pin differs: {name}")
        pins[sidecar.relative_to(root).as_posix()] = record["manifest_sha256"]
        if record["sha256"] not in frames:
            frames[record["sha256"]] = load_canonical_artifact(path, expected_type="memberships", allow_research=True)[0]
    return list(frames.values())


def _totals(proofs: pd.DataFrame, rejections: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    def counts(series: pd.Series) -> dict[str, int]:
        return {str(key): int(value) for key, value in sorted(series.items())}

    def days(start: pd.Series, end: pd.Series) -> pd.Series:
        return ((end.fillna(cutoff).clip(upper=cutoff) - start.clip(upper=cutoff)).dt.total_seconds() / 86_400).clip(lower=0)

    # Proven spells whose target rows leave part of them uncovered; that part is neither proven nor a rejected spell.
    spell = ["source_security_id", "ticker", "legacy_spell_from_utc"]
    covered = days(proofs.effective_from_utc, proofs.effective_to_utc).groupby([proofs[key] for key in spell]).sum()
    spells = proofs.drop_duplicates(spell)
    total = days(spells.legacy_spell_from_utc, spells.legacy_spell_to_utc).set_axis(pd.MultiIndex.from_frame(spells[spell]))
    remainder = (total - covered.reindex(total.index)).round(9)
    return {"proof_rows": len(proofs), "proven_query_ids": int(proofs.source_security_id.nunique()),
            "proven_spells_with_unproven_remainder": int(remainder.gt(0).sum()),
            "unproven_spell_remainder_days_until_target_cutoff": float(remainder.clip(lower=0).sum()),
            "proven_query_ids_by_kind": counts(proofs.groupby("proof_kind").source_security_id.nunique()),
            "rejected_spells_by_reason": counts(rejections.groupby("reason").size()),
            "rejected_query_ids_by_reason": counts(rejections.groupby("reason").source_security_id.nunique()),
            "partially_proven_query_ids": len(set(proofs.source_security_id) & set(rejections.source_security_id)),
            "target_securities": int(proofs.target_security_id.nunique())}


def publish_legacy_query_identity_proofs(*, root: Path, config: Path, config_sha256: str, output: Path) -> dict[str, Any]:
    """Prove or reject every legacy query identity the CIK bridge left unconverted; never admission."""
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    _require(output.parent == root / "data/research", "legacy identity proofs must be a new direct child of data/research")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("prove-legacy-query-identities", runtime_dir=runtime, config_path=config):
        _guard()
        if output.exists():
            raise FileExistsError(f"immutable legacy identity proofs already exist: {output.relative_to(root).as_posix()}")
        pins: dict[str, str] = {}
        relative_config = config.relative_to(root).as_posix()
        settings = pinned_object(pin_file(root, {"path": relative_config, "sha256": config_sha256}, pins), config_sha256)
        _require(set(settings) == _KEYS and settings["schema"] == CONFIG_SCHEMA, "legacy identity proof configuration differs")
        bridge, bridge_sources = load_identity_bridge(root, settings["identity_manifest"], pins)
        memberships, cutoff, lineage = _target(root, settings["target_membership_authority"], bridge_sources, pins)
        changes = _changes(root, settings, lineage, pins)
        corrected = _corrected(root, settings["symbol_corrections"], bridge_sources, pins)
        legacy = _legacy(root, settings, pins)
        transitions = pd.read_parquet(pin_file(root, settings["security_transitions"], pins))
        _guard()
        evidence = json_sha256(dict(sorted(pins.items())))
        proofs, rejections = build_legacy_query_proofs(legacy, memberships, changes, transitions, bridge,
            corrected_security_ids=corrected, target_cutoff_utc=cutoff, evidence_sha256=evidence)
        _guard()
        staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
        staging.mkdir()
        try:
            proofs.to_parquet(staging / "proofs.parquet", index=False)
            rejections.to_parquet(staging / "rejections.parquet", index=False)
            package = Path(__file__).resolve().parents[1]
            report = {
                "schema": SCHEMA, "status": "complete", "config": {"path": relative_config, "sha256": config_sha256},
                "source_files": dict(sorted(pins.items())),
                "implementation_files": {f"market_predictor/{name}": file_sha256(package / name)
                                         for name in IMPLEMENTATION_PATHS},
                "evidence_sha256": evidence, "target_cutoff_utc": cutoff.isoformat(),
                "proofs_sha256": file_sha256(staging / "proofs.parquet"),
                "rejections_sha256": file_sha256(staging / "rejections.parquet"),
                "totals": _totals(proofs, rejections, cutoff), "strength_order": list(PROOF_KINDS),
                "kind_evidence": KIND_EVIDENCE,
                "availability_basis": AVAILABILITY_BASIS, "evidence_complete_date": EVIDENCE_COMPLETE_DATE,
                "transition_evidence": TRANSITION_EVIDENCE, "symbol_validity": SYMBOL_VALIDITY,
                "scope": "legacy IDs that are neither CIK-bridge sources nor target IDs; targets are the global membership "
                         "authority, and cohort membership is decided by consumers",
                "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
            }
            write_json_object(staging / "_manifest.json", report)
            _guard()
            staging.rename(output)
        finally:
            if staging.exists():
                for name in ("proofs.parquet", "rejections.parquet", "_manifest.json"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}
