"""Proof publication over a target authority built by the real S&P membership publisher."""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import legacy_query_identity_proofs as publisher
from market_predictor.universe.sp500 import membership_authority as authority_module
from market_predictor.universe.sp500.membership_history import (
    IndexChange,
    IndexChangeSource,
    VerifiedIndexChanges,
    _security_identity_for_interval,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FISERV = "cik:0000798354"
EVENTS = "data/canonical/index_membership/events"
RAW = "data/raw/index_membership/official"


def _t(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="America/New_York").tz_convert("UTC")


def _change(action: str, ticker: str, company: str, day: str) -> IndexChange:
    source = IndexChangeSource(source_url=f"https://press.spglobal.com/{ticker}-{action}",
                               source_published_date=date(2019, 1, 2), source_sha256="b" * 64)
    return IndexChange(effective_at_utc=_t(day).to_pydatetime(), action=action, ticker=ticker, company=company,
                       sector="Industrials", source_url=source.source_url, source_published_date=source.source_published_date,
                       source_sha256=source.source_sha256, supporting_sources=(source,))


CHANGES = (_change("addition", "T000", "Company 0", "2021-06-01"), _change("deletion", "OLDA", "Old A Inc", "2021-06-01"),
           _change("addition", "SPEL", "Spell Co", "2020-01-02"), _change("deletion", "OLDC", "Old C Corp", "2020-01-02"),
           _change("deletion", "SPEL", "Spell Company Inc", "2021-01-04"), _change("addition", "T001", "Company 1", "2021-01-04"))


def _json(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.as_posix(), "sha256": file_sha256(path)}


def _audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic", status="pass", failures=0, rows_checked=rows,
                                                            detail="test-only artifact"),))


def _target(root: Path, patch: pytest.MonkeyPatch) -> tuple[Path, pd.DataFrame]:
    """The real membership publisher over a 500-name anchor whose T020 is Fiserv's corrected CIK."""
    for name, payload in ((RAW, {"artifact_sha256": "a" * 64}), (EVENTS, {"event_set_sha256": "b" * 64}),
                          ("data/canonical/index_membership/transitions", {"transition_set_sha256": "c" * 64})):
        (root / name).mkdir(parents=True)
        authority_module._write_json_atomic(root / name / "_authority.json", payload)
    anchor = pd.DataFrame({"ticker": [f"T{number:03d}" for number in range(500)],
                           "company": [f"Company {number}" for number in range(500)], "sector": ["Industrials"] * 500,
                           "industry": ["Machinery"] * 500, "cik": [f"{number + 1:010d}" for number in range(500)]})
    anchor.loc[20, "cik"] = "0000798354"
    anchor.to_csv(root / "anchor.csv", index=False)
    (root / "reviewed.csv").write_text("bound input\n", encoding="utf-8")
    verified = VerifiedIndexChanges(changes=CHANGES, authority_sha256=file_sha256(root / EVENTS / "_authority.json"),
                                    event_set_sha256="b" * 64)
    empty = pd.DataFrame({column: pd.Series(dtype=kind) for column, kind in (
        ("transition_id", "str"), ("effective_at_utc", "datetime64[ns, UTC]"), ("old_ticker", "str"), ("new_ticker", "str"),
        ("identity_continuity", "bool"), ("membership_continuity", "bool"), ("old_security_id", "str"),
        ("new_security_id", "str"), ("source_url", "str"))})
    patch.setattr(authority_module, "require_sp500_transition_authority", lambda *_, **__: empty)
    patch.setattr(authority_module, "require_spglobal_event_reconstruction_ready", lambda *_, **__: verified)
    directory = root / "data/canonical/index_membership/memberships"
    authority_module.publish_sp500_membership_authority(archive_directory=root / RAW, event_directory=root / EVENTS,
        transition_directory=root / "data/canonical/index_membership/transitions", reviewed_transitions_path=root / "reviewed.csv",
        anchor_path=root / "anchor.csv", start_date=date(2018, 5, 29), cutoff_date=date(2026, 7, 8), output_directory=directory)
    frame, _ = authority_module.load_sp500_membership_authority_envelope(directory)
    return directory / "_authority.json", frame


def _legacy_row(template: pd.Series, security: str, ticker: str, start: str, end: str | None = None) -> dict[str, Any]:
    return {**template.to_dict(), "security_id": security, "ticker": ticker, "effective_from_utc": _t(start),
            "effective_to_utc": _t(end) if end else pd.NaT, "available_at_utc": _t(start)}


def _legacy_id(ticker: str, company: str, start: str, end: str) -> str:
    return _security_identity_for_interval(ticker=ticker, company=company, effective_from=_t(start), effective_to=_t(end),
                                           current=pd.DataFrame(), aliases=[])


def _memberships(root: Path, name: str, rows: list[dict[str, Any]], columns: list[str]) -> dict[str, str]:
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    path = root / f"data/canonical/{name}.parquet"
    write_canonical_artifact(frame, path, artifact_type="memberships", audit=_audit(len(frame)), production_ready=False)
    return {"path": f"data/canonical/{name}.parquet", "sha256": file_sha256(path),
            "manifest_sha256": file_sha256(manifest_path_for(path))}


def _world(root: Path, patch: pytest.MonkeyPatch) -> dict[str, Any]:
    root.mkdir(parents=True)
    authority, target = _target(root, patch)
    template = target.iloc[0]
    shared = [_legacy_row(template, _legacy_id("OLDA", "Old A Inc", "2019-07-09", "2021-06-01"), "OLDA", "2019-07-09", "2021-06-01"),
              _legacy_row(template, _legacy_id("SPEL", "Spell Co", "2020-01-02", "2021-01-04"), "SPEL", "2020-01-02", "2021-01-04"),
              _legacy_row(template, "cik:0000000005:ticker:T004", "T004", "2019-07-09"),
              _legacy_row(template, "cusip:AAA111111", "X01", "2019-07-09", "2020-06-01"),
              _legacy_row(template, "cusip:AAA111111", "T010", "2020-06-01"),
              _legacy_row(template, "cik:0000000010:ticker:T009", "T009", "2019-07-09"),
              _legacy_row(template, "cik:0000000031", "T030", "2019-07-09"),
              _legacy_row(template, "cusip:CCC333333", "ZZZ", "2019-07-09", "2020-01-02")]
    columns = list(target.columns)
    early = _memberships(root, "legacy_verified", shared, columns)
    later = _memberships(root, "legacy_original", [*shared, _legacy_row(template, "cusip:BBB222222", "T020", "2019-07-09")], columns)
    transitions = root / "data/raw/index_membership/transitions.parquet"
    pd.DataFrame([{"id": "x01-t010", "old_symbol": "X01", "new_symbol": "T010", "old_cusip": "AAA111111",
                   "new_cusip": "AAA111111", "identity_continuity": True, "effective_date": "2020-06-01",
                   "process_date": "2020-06-01"}]).to_parquet(transitions, index=False)
    bridge = pd.DataFrame([{"source_security_id": "cik:0000000010:ticker:T009", "ticker": "T009",
                            "target_security_id": "cik:0000000010", "effective_from_utc": _t("2019-07-09"),
                            "effective_to_utc": pd.NaT, "available_at_utc": _t("2019-07-09"), "bridge_row_sha256": "d" * 64}])
    bridge["effective_to_utc"] = pd.to_datetime(bridge.effective_to_utc, utc=True)
    bridge_path = root / "data/research/identity/identity_bridge.parquet"
    write_canonical_artifact(bridge, bridge_path, artifact_type="issuer_news_identity_bridge", audit=_audit(1),
                             production_ready=False)
    corrections = root / "configs/swing_symbol_corrections.toml"
    corrections.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPOSITORY / "configs/swing_symbol_corrections.toml", corrections)
    files = {path.relative_to(root).as_posix(): file_sha256(path)
             for path in (bridge_path, manifest_path_for(bridge_path), authority, corrections)}
    identity = _json(root / "data/research/identity/_manifest.json",
                     {"schema": "market_predictor.issuer_news_identity_alignment", "source_files": files})
    sources: dict[str, Any] = {"corrected": {"kind": "corrected"}}
    archives = {}
    for name, memberships in (("early", early), ("later", later)):
        body = {"schema": "test.alpaca_news_request", "memberships_sha256": memberships["sha256"]}
        collection = root / f"data/raw/{name}_news"
        _json(collection / "_request.json", {**body, "request_sha256": json_sha256(body)})
        raw = _json(collection / "_manifest.json", {"request_sha256": json_sha256(body)})
        derived = _json(root / f"data/research/{name}_derived/_manifest.json",
                        {"request": {"collection_manifest_sha256": raw["sha256"]}})
        sources[name] = {"kind": "derived", "directory": f"data/research/{name}_derived", "manifest_sha256": derived["sha256"]}
        archives[name] = {"collection": f"data/raw/{name}_news", "memberships": memberships}
    monthly = _json(root / "configs/monthly.json", {"sources": sources})
    config = {"schema": publisher.CONFIG_SCHEMA, "identity_manifest": _relative(root, identity),
              "target_membership_authority": _pin(root, authority),
              "symbol_corrections": _pin(root, corrections), "event_authority": _pin(root, root / EVENTS / "_authority.json"),
              "event_raw_archive": _pin(root, root / RAW / "_authority.json"), "security_transitions": _pin(root, transitions),
              "monthly_news_config": _relative(root, monthly), "archives": archives}
    path = root / "configs/proofs.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    verified = VerifiedIndexChanges(changes=CHANGES, authority_sha256=file_sha256(root / EVENTS / "_authority.json"),
                                    event_set_sha256="b" * 64)
    return {"root": root, "config": path, "sha256": file_sha256(path), "settings": config, "verified": verified}


def _pin(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}


def _relative(root: Path, record: dict[str, str]) -> dict[str, str]:
    return {"path": Path(record["path"]).relative_to(root).as_posix(), "sha256": record["sha256"]}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        return _world(tmp_path_factory.mktemp("proofs") / "repo", patch)


@pytest.fixture(autouse=True)
def _policy(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    calls: list[tuple[Path, Path]] = []
    verified: VerifiedIndexChanges = world["verified"]

    def verify(event: Path, *, archive_directory: Path) -> VerifiedIndexChanges:
        calls.append((event, archive_directory))
        return verified

    monkeypatch.setattr(publisher, "_guard", lambda: None)
    monkeypatch.setattr(publisher, "require_spglobal_event_reconstruction_ready", verify)
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)
    return calls


def _run(world: dict[str, Any], name: str, *, config: Path | None = None, sha256: str | None = None) -> dict[str, Any]:
    return publisher.publish_legacy_query_identity_proofs(root=world["root"], config=config or world["config"],
        config_sha256=sha256 or world["sha256"], output=world["root"] / "data/research" / name)


def test_publishes_every_kind_with_relative_pins_and_is_immutable(world: dict[str, Any],
                                                                  _policy: list[tuple[Path, Path]]) -> None:
    report = _run(world, "proofs")
    root = world["root"]
    assert _policy == [(root / EVENTS, root / RAW)]
    assert report["totals"]["proven_query_ids_by_kind"] == {"cik_equal": 1, "company_ticker_hash_reproduced": 1,
        "cusip_chain_end_ticker_match": 1, "sp500_spell_events_reproduced": 1}
    assert report["totals"]["rejected_query_ids_by_reason"] == {"corrected_security_uses_corrections_archive_only": 1,
                                                                "no_candidate": 1}
    assert report["totals"]["proven_spells_with_unproven_remainder"] == 0
    assert report["totals"]["unproven_spell_remainder_days_until_target_cutoff"] == 0.0
    proofs = pd.read_parquet(root / "data/research/proofs/proofs.parquet")
    rejections = pd.read_parquet(root / "data/research/proofs/rejections.parquet")
    assert set(proofs.source_security_id).isdisjoint({"cik:0000000010:ticker:T009", "cik:0000000031"})
    chain = proofs.loc[proofs.source_security_id.eq("cusip:AAA111111")]
    assert chain.target_security_id.eq("cik:0000000011").all() and chain.ticker.tolist() == ["X01", "T010"]
    corrected = rejections.loc[rejections.source_security_id.eq("cusip:BBB222222")]
    assert corrected.reason.item() == "corrected_security_uses_corrections_archive_only"
    assert json.loads(corrected.detail_json.item()) == {"target": FISERV}
    manifest = json.loads((root / "data/research/proofs/_manifest.json").read_text())
    assert manifest["proofs_sha256"] == file_sha256(root / "data/research/proofs/proofs.parquet")
    assert {"data/raw/early_news/_request.json", "data/raw/later_news/_request.json", "data/canonical/legacy_original.parquet",
            f"{EVENTS}/_authority.json", f"{RAW}/_authority.json"} <= set(manifest["source_files"])
    assert not any(Path(path).is_absolute() or "\\" in path for path in manifest["source_files"])
    assert "market_predictor/universe/sp500/membership_history.py" in manifest["implementation_files"]
    assert manifest["availability_basis"] == "retrospective_membership_effective_proxy"
    assert "no collection receipt" in manifest["transition_evidence"] and manifest["evidence_complete_date"]
    assert manifest["evidence_sha256"] == json_sha256(manifest["source_files"])
    assert not any(manifest[flag] for flag in ("training_eligible", "serving_eligible", "promotion_eligible"))
    loaded, _ = publisher.load_legacy_query_proofs(root, {"path": "data/research/proofs/_manifest.json",
                                                          "sha256": report["manifest_sha256"]}, {})
    pd.testing.assert_frame_equal(loaded, proofs)
    with pytest.raises(FileExistsError):
        _run(world, "proofs")
    _run(world, "proofs-again")
    pd.testing.assert_frame_equal(pd.read_parquet(root / "data/research/proofs-again/proofs.parquet"), proofs)


@pytest.mark.parametrize("target", ["configs/proofs.json", "data/research/identity/_manifest.json",
    "data/research/identity/identity_bridge.parquet", "data/research/identity/identity_bridge.parquet.manifest.json",
    "data/canonical/index_membership/memberships/_authority.json", "data/canonical/index_membership/memberships/_manifest.json",
    "configs/swing_symbol_corrections.toml", f"{EVENTS}/_authority.json", f"{RAW}/_authority.json",
    "data/raw/index_membership/transitions.parquet", "configs/monthly.json", "data/research/early_derived/_manifest.json",
    "data/raw/later_news/_manifest.json", "data/canonical/legacy_original.parquet",
    "data/canonical/legacy_verified.parquet.manifest.json"])
def test_every_pin_is_verified_before_any_output(world: dict[str, Any], target: str) -> None:
    path, output = world["root"] / target, world["root"] / "data/research" / f"tamper-{Path(target).name}"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(DataReadinessError):
            _run(world, output.name)
    finally:
        path.write_bytes(original)
    assert not output.exists()


def _variant(world: dict[str, Any], name: str, **changes: Any) -> tuple[Path, str]:
    path = world["root"] / f"configs/{name}.json"
    path.write_text(json.dumps({**world["settings"], **changes}), encoding="utf-8")
    return path, file_sha256(path)


def test_lineage_must_bind_every_source_to_the_bridge_and_target(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    root, settings = world["root"], world["settings"]
    other = root / "data/canonical/index_membership/other_events/_authority.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({"event_set_sha256": "b" * 64, "other": True}), encoding="utf-8")
    swapped = {"early": {**settings["archives"]["early"], "memberships": settings["archives"]["later"]["memberships"]},
               "later": settings["archives"]["later"]}
    for name, changes, message in (
            ("events", {"event_authority": _pin(root, other)}, "not the target membership authority's parent"),
            ("swapped", {"archives": swapped}, "not minted from the pinned membership file"),
            ("missing", {"archives": {"early": settings["archives"]["early"]}}, "every derived news archive"),
            ("extra", {**settings, "unexpected": True}, "configuration differs")):
        config, sha256 = _variant(world, name, **changes)
        with pytest.raises(DataReadinessError, match=message):
            _run(world, f"lineage-{name}", config=config, sha256=sha256)
    identity = json.loads((root / settings["identity_manifest"]["path"]).read_text())
    identity["source_files"].pop("configs/swing_symbol_corrections.toml")
    (root / "data/research/unbound_identity").mkdir()
    for name in ("identity_bridge.parquet", "identity_bridge.parquet.manifest.json"):
        shutil.copyfile(root / "data/research/identity" / name, root / "data/research/unbound_identity" / name)
    identity["source_files"] = {key.replace("/identity/", "/unbound_identity/"): value
                                for key, value in identity["source_files"].items()}
    unbound = _json(root / "data/research/unbound_identity/_manifest.json", identity)
    config, sha256 = _variant(world, "unbound", identity_manifest=_relative(root, unbound))
    with pytest.raises(DataReadinessError, match="symbol corrections differ from the CIK bridge's"):
        _run(world, "lineage-unbound", config=config, sha256=sha256)
    changed = VerifiedIndexChanges(changes=CHANGES, authority_sha256=world["verified"].authority_sha256,
                                   event_set_sha256="0" * 64)
    monkeypatch.setattr(publisher, "require_spglobal_event_reconstruction_ready", lambda *_, **__: changed)
    with pytest.raises(DataReadinessError, match="verified event authority differs"):
        _run(world, "lineage-event-set")
    assert not any((root / "data/research").glob("lineage-*"))


def test_busy_lease_and_output_location_prevent_any_source_read(world: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publisher, "load_identity_bridge", lambda *_: pytest.fail("source read without the lease"))
    with heavy_job_lease("test-owner", runtime_dir=world["root"] / "data/runtime"), pytest.raises(HeavyJobBusyError):
        _run(world, "busy")
    with pytest.raises(DataReadinessError, match="direct child of data/research"):
        publisher.publish_legacy_query_identity_proofs(root=world["root"], config=world["config"],
            config_sha256=world["sha256"], output=world["root"] / "data/research/nested/proofs")
    assert not (world["root"] / "data/research/busy").exists()


def test_published_proofs_are_verified_when_loaded(world: dict[str, Any]) -> None:
    report = _run(world, "loaded")
    proofs = world["root"] / "data/research/loaded/proofs.parquet"
    record = {"path": "data/research/loaded/_manifest.json", "sha256": report["manifest_sha256"]}
    original = proofs.read_bytes()
    try:
        proofs.write_bytes(original + b" ")
        with pytest.raises(DataReadinessError, match="differ from their manifest"):
            publisher.load_legacy_query_proofs(world["root"], record, {})
    finally:
        proofs.write_bytes(original)
    with pytest.raises(DataReadinessError, match="pin differs"):
        publisher.load_legacy_query_proofs(world["root"], {**record, "sha256": "0" * 64}, {})
