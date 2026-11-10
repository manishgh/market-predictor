"""Receipt semantics are checked beyond hashes; inherited files are never rewritten."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import investment_label_reuse as module
from market_predictor.research.investment_features import FLAGS
from market_predictor.swing.contracts.holding_materialization import SourcePin


def write(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    package = tmp_path / "current/src/market_predictor"
    old_root = tmp_path / "snapshots" / ("a" * 40)
    old_package = old_root / "src/market_predictor"
    names = {"investment/contracts.py": "core", "research/investment_labels.py": "orchestration",
        "commands/investment_research.py": "cli", "research/investment_label_reuse.py": "reuse",
        "research_api/static/app.js": "script", "research_api/static/index.css": "stylesheet",
        "research_api/static/index.html": "page"}
    exported = {}
    current = {}
    for name, body in names.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        if name.endswith(".py"):
            current[str(path)] = file_sha256(path)
        if name != "research/investment_label_reuse.py":
            old = old_package / name
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text("original " + body if name in module.REVIEWED_IMPLEMENTATION_CHANGES else body)
            exported["src/market_predictor/" + name] = file_sha256(old)
    monkeypatch.setattr(module, "__file__", str(package / "research/investment_label_reuse.py"))
    monkeypatch.setattr(module, "guard", lambda *_: None)
    export_pin = write(old_root / "_export.json", {"commit": "a" * 40, "files": exported, "git_archive_sha256": "b" * 64})
    consumer_export = write(package.parent.parent / "_export.json", {"commit": "d" * 40,
        "files": {"src/market_predictor/" + path.relative_to(package).as_posix(): file_sha256(path)
            for path in package.rglob("*") if path.is_file()}, "git_archive_sha256": "e" * 64})
    parent = tmp_path / "parent"
    original_impl = {str(old_package / name): file_sha256(old_package / name)
        for name in names if name not in module.RUNTIME_ADDITIONS and name.endswith(".py")}
    request = write(tmp_path / "request.json", {"frozen": True})
    source = write(tmp_path / "source.json", {"actual": "fixture source"})
    identity = json_sha256({"request_sha256": request["sha256"], "implementation": original_impl})
    clock = "2026-11-10T01:47:45.260786+00:00"
    job = write(parent / "_job.json", {"job_identity": identity, "verified_at_utc": clock})
    common_value = {**original_impl, request["path"]: request["sha256"], source["path"]: source["sha256"],
        job["path"]: job["sha256"]}
    common = write(parent / "_verified_sources.json", common_value)
    row = {"decision_id": "decision-one", "security_id": "issuer:one", "decision_session": "2023-05-31",
        "ticker": "ONE", "sector_benchmark": "XLK", "status": "eligible"}
    records = [{"decision_id": row["decision_id"], "security_id": row["security_id"],
        "decision_session": row["decision_session"], "horizon_sessions": h, "roster_status": "eligible",
        "membership_verified_at_utc": clock, "components": [{"sources": [
            {"reference": source["path"], "artifact_sha256": source["sha256"]}]}], **FLAGS} for h in (63, 252)]
    directory = parent / "securities" / json_sha256(row["security_id"])
    labels = write(directory / "labels.json", {"rows": records})
    checkpoint_value = {"schema": "market_predictor.investment_security_labels", "status": "security_complete",
        "security_id": row["security_id"], "job_identity": identity, "declared_rows": 2,
        "files": {"labels.json": labels["sha256"]}, "source_files": {item["path"]: item["sha256"]
            for item in (request, source, job, common)}, **FLAGS}
    checkpoint = write(directory / "_manifest.json", checkpoint_value)
    entry = {"security_id": row["security_id"], "checkpoint": checkpoint, "labels": labels,
        "declared_rows": 2, "row_keys_sha256": module.row_keys_sha256(records)}
    receipt = {"schema": "market_predictor.investment_label_parent_receipt", "request": request, "parent_job": job,
        "parent_common_sources": common, "parent_implementation_export": export_pin,
        "consumer_implementation_export": consumer_export,
        "parent_job_identity": identity, "parent_verified_at_utc": clock,
        "declared_checkpoint_count": 1, "declared_rows": 2, "security_checkpoints": [entry]}
    receipt_path = tmp_path / "receipt.json"
    receipt_pin = write(receipt_path, receipt)
    output = tmp_path / "new-output"
    new_job = write(output / "_job.json", {"job_identity": "c" * 64, "verified_at_utc": "2026-11-10T09:00:00+00:00"})
    kwargs = {"root": tmp_path, "receipt_pin": SourcePin(**receipt_pin), "request": SourcePin(**request),
        "groups": {row["security_id"]: [row]}, "current_base_sources": {**current,
            request["path"]: request["sha256"], source["path"]: source["sha256"], new_job["path"]: new_job["sha256"]},
        "current_implementation": current, "current_job": Path(new_job["path"]), "output": output}
    return {"kwargs": kwargs, "receipt": receipt, "receipt_path": receipt_path, "entry": entry,
        "checkpoint_value": checkpoint_value, "records": records, "source": source, "package": package,
        "old_root": old_root, "row": row, "parent": parent, "common_value": common_value}


def repin(value: dict[str, Any], *, records: bool = False, checkpoint: bool = False) -> None:
    entry = value["entry"]
    if records:
        entry["labels"] = write(Path(entry["labels"]["path"]), {"rows": value["records"]})
        value["checkpoint_value"]["files"]["labels.json"] = entry["labels"]["sha256"]
        checkpoint = True
    if checkpoint:
        entry["checkpoint"] = write(Path(entry["checkpoint"]["path"]), value["checkpoint_value"])
    value["kwargs"]["receipt_pin"] = SourcePin(**write(value["receipt_path"], value["receipt"]))


def test_valid_parent_preserves_bytes_and_explicit_original_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in value["parent"].rglob("*") if p.is_file()}
    parent = module.verify_parent_checkpoints(**value["kwargs"])
    origin = parent.origin(parent.entries["issuer:one"]).model_dump(mode="json")
    assert origin["origin"] == "inherited" and origin["producer_commit"] == "a" * 40
    assert origin["job_identity"] == value["receipt"]["parent_job_identity"]
    assert datetime.fromisoformat(origin["verified_at_utc"]) == datetime.fromisoformat(value["receipt"]["parent_verified_at_utc"])
    assert origin["checkpoint"] == value["entry"]["checkpoint"]
    assert origin["labels"] == value["entry"]["labels"]
    assert origin["common_sources"] == value["receipt"]["parent_common_sources"]
    assert all(Path(path).read_bytes() == data for path, data in before.items())
    assert all(str(path) in parent.source_files for path in (value["package"]).rglob("*.py")
        if path.name == "contracts.py")


@pytest.mark.parametrize("poison", ["request", "job", "clock", "core", "producer", "base_extra", "base_missing",
    "base_hash", "missing_parent", "extra_parent", "duplicate_entry", "wrong_security_path", "row_keys",
    "row_missing", "row_duplicate", "row_extra", "row_security", "row_session", "row_clock", "row_source", "flags",
    "local_source", "duplicate_binding", "consumer_path", "consumer_hash", "consumer_producer",
    "current_asset", "parent_asset", "unexpected_source"])
def test_reuse_refuses_pinned_semantic_or_byte_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str
) -> None:
    value = fixture(tmp_path, monkeypatch)
    entry, receipt = value["entry"], value["receipt"]
    kwargs = value["kwargs"]
    if poison == "request":
        kwargs["request"] = SourcePin(path=receipt["request"]["path"], sha256="f" * 64)
    elif poison == "consumer_path":
        receipt["consumer_implementation_export"] = write(tmp_path / "another-export.json",
            json.loads(Path(receipt["consumer_implementation_export"]["path"]).read_text()))
    elif poison == "consumer_hash":
        receipt["consumer_implementation_export"]["sha256"] = "f" * 64
    elif poison == "consumer_producer":
        path = Path(receipt["consumer_implementation_export"]["path"])
        data = json.loads(path.read_text())
        data["commit"] = "z" * 40
        receipt["consumer_implementation_export"] = write(path, data)
    elif poison == "current_asset":
        (value["package"] / "research_api/static/app.js").write_text("changed script")
    elif poison == "parent_asset":
        (value["old_root"] / "src/market_predictor/research_api/static/index.css").write_text("changed old stylesheet")
    elif poison == "unexpected_source":
        (value["package"] / "unexpected.py").write_text("unlisted code")
    elif poison == "job":
        receipt["parent_job_identity"] = "f" * 64
    elif poison == "clock":
        receipt["parent_verified_at_utc"] = "2026-11-10T01:47:46.260786+00:00"
    elif poison == "core":
        (value["package"] / "investment/contracts.py").write_text("changed current core")
    elif poison == "producer":
        (value["old_root"] / "src/market_predictor/investment/contracts.py").write_text("changed original core")
    elif poison == "base_extra":
        kwargs["current_base_sources"][str(tmp_path / "unrecorded.json")] = "f" * 64
    elif poison == "base_missing":
        kwargs["current_base_sources"].pop(value["source"]["path"])
    elif poison == "base_hash":
        kwargs["current_base_sources"][value["source"]["path"]] = "f" * 64
    elif poison == "missing_parent":
        Path(entry["checkpoint"]["path"]).unlink()
    elif poison == "extra_parent":
        write(value["parent"] / "securities/extra/_manifest.json", {})
    elif poison == "duplicate_entry":
        receipt["security_checkpoints"].append(dict(entry))
        receipt["declared_checkpoint_count"] = 2
        receipt["declared_rows"] = 4
    elif poison == "wrong_security_path":
        entry["security_id"] = "issuer:other"
        kwargs["groups"]["issuer:other"] = kwargs["groups"].pop("issuer:one")
    elif poison == "row_keys":
        entry["row_keys_sha256"] = "f" * 64
    elif poison == "row_missing":
        value["records"].pop()
    elif poison == "row_duplicate":
        value["records"][1] = value["records"][0]
    elif poison == "row_extra":
        value["records"].append({**value["records"][0], "decision_id": "extra"})
    elif poison == "row_security":
        value["records"][0]["security_id"] = "issuer:other"
    elif poison == "row_session":
        value["records"][0]["decision_session"] = "2023-06-01"
    elif poison == "row_clock":
        value["records"][0]["membership_verified_at_utc"] = "2026-11-10T01:47:46.260786+00:00"
    elif poison == "row_source":
        value["records"][0]["components"][0]["sources"][0]["artifact_sha256"] = "f" * 64
    elif poison == "flags":
        value["records"][0]["training_ready"] = True
    elif poison == "local_source":
        value["checkpoint_value"]["source_files"].pop(receipt["parent_job"]["path"])
    elif poison == "duplicate_binding":
        value["row"]["sector_benchmark"] = "SPY"
    repin(value, records=(poison.startswith("row_") and poison != "row_keys") or poison == "flags",
        checkpoint=poison == "local_source")
    with pytest.raises((DataReadinessError, OSError)):
        module.verify_parent_checkpoints(**kwargs)


def test_source_tamper_is_fatal_even_when_both_inventories_still_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = fixture(tmp_path, monkeypatch)
    Path(value["source"]["path"]).write_text("tampered source after collection")
    with pytest.raises(DataReadinessError, match="source bytes changed"):
        module.verify_parent_checkpoints(**value["kwargs"])


@pytest.mark.parametrize("disappear_after_verification", [False, True])
def test_real_consumer_integration_reuses_original_paths_jobs_clocks_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disappear_after_verification: bool
) -> None:
    from market_predictor.research import investment_labels as labels
    from tests.test_investment_labels import setup_job

    value = fixture(tmp_path / "producer-fixture", monkeypatch)
    # Old run predates the consumer receipt; its producer has its own original pins.
    current_export_path = value["package"].parent.parent / "_export.json"
    current_export_data = json.loads(current_export_path.read_text())
    current_export_path.unlink()
    request, digest, parent_output = setup_job(tmp_path, monkeypatch)
    old_package = value["old_root"] / "src/market_predictor"
    original_impl = {str(old_package / name): file_sha256(old_package / name)
        for name in ("investment/contracts.py", "research/investment_labels.py")}
    current_impl = value["kwargs"]["current_implementation"]
    monkeypatch.setattr(labels, "_implementation", lambda: original_impl)
    labels.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest, output=parent_output)
    checkpoint_paths = sorted((parent_output / "securities").glob("*/_manifest.json"))
    # Retain one parent checkpoint, as in a stopped serial batch; build only the other.
    inherited_path = next(path for path in checkpoint_paths
        if json.loads(path.read_text())["security_id"] == "issuer:one")
    other = next(path.parent for path in checkpoint_paths if path != inherited_path)
    for path in other.iterdir():
        path.unlink()
    other.rmdir()
    entry_manifest = json.loads(inherited_path.read_text())
    records = json.loads((inherited_path.parent / "labels.json").read_text())["rows"]
    job_path = parent_output / "_job.json"
    job = json.loads(job_path.read_text())
    receipt = {"schema": "market_predictor.investment_label_parent_receipt",
        "request": {"path": str(request), "sha256": digest},
        "parent_job": {"path": str(job_path), "sha256": file_sha256(job_path)},
        "parent_common_sources": {"path": str(parent_output / "_verified_sources.json"),
            "sha256": file_sha256(parent_output / "_verified_sources.json")},
        "parent_implementation_export": value["receipt"]["parent_implementation_export"],
        "consumer_implementation_export": value["receipt"]["consumer_implementation_export"],
        "parent_job_identity": job["job_identity"], "parent_verified_at_utc": job["verified_at_utc"],
        "declared_checkpoint_count": 1, "declared_rows": 2, "security_checkpoints": [{
            "security_id": "issuer:one", "checkpoint": {"path": str(inherited_path), "sha256": file_sha256(inherited_path)},
            "labels": {"path": str(inherited_path.parent / "labels.json"), "sha256": entry_manifest["files"]["labels.json"]},
            "declared_rows": 2, "row_keys_sha256": module.row_keys_sha256(records)}]}
    receipt_pin = SourcePin(**write(tmp_path / "inheritance.json", receipt))
    before = {str(path): path.read_bytes() for path in parent_output.rglob("*") if path.is_file()}
    package = value["package"]
    write(package.parent.parent / "_export.json", current_export_data)
    monkeypatch.setattr(labels, "_implementation", lambda: current_impl)
    monkeypatch.setattr(labels, "compile_investment_label", lambda **_: pytest.fail("inherited eligible rows recomputed"))
    output = tmp_path / "continued"
    if disappear_after_verification:
        verify = labels.verify_parent_checkpoints

        def verified_then_removed(**kwargs: Any) -> Any:
            result = verify(**kwargs)
            for path in inherited_path.parent.iterdir():
                path.unlink()
            inherited_path.parent.rmdir()
            return result

        monkeypatch.setattr(labels, "verify_parent_checkpoints", verified_then_removed)
        monkeypatch.setattr(labels, "_publish", lambda *_args, **_kwargs: pytest.fail("missing parent was republished"))
        with pytest.raises(OSError):
            labels.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest,
                output=output, parent_receipt=receipt_pin)
        assert not inherited_path.parent.exists()
        return
    labels.materialize_investment_labels(root=tmp_path, request=request, request_sha256=digest,
        output=output, parent_receipt=receipt_pin)
    report = json.loads((output / "final/_manifest.json").read_text())
    assert report["inherited_security_checkpoint_count"] == report["new_security_checkpoint_count"] == 1
    assert report["inherited_label_rows"] == report["new_label_rows"] == 2
    assert report["label_rows"] == 4
    origins = {item["security_id"]: item for item in report["security_checkpoint_origins"]}
    old_origin, new_origin = origins["issuer:one"], origins["issuer:two"]
    assert old_origin["checkpoint"]["path"] == str(inherited_path)
    assert old_origin["job_identity"] == job["job_identity"] != new_origin["job_identity"]
    assert datetime.fromisoformat(old_origin["verified_at_utc"]) == datetime.fromisoformat(job["verified_at_utc"])
    assert old_origin["producer_commit"] == "a" * 40 and new_origin["producer_commit"] == "d" * 40
    assert report["parent_receipt"] == receipt_pin.model_dump(mode="json")
    expected_job = json_sha256({"request_sha256": digest, "implementation": current_impl,
        "parent_receipt": receipt_pin.model_dump(mode="json")})
    assert report["job_identity"] == expected_job
    assert not (output / "securities" / json_sha256("issuer:one")).exists()
    assert all(Path(name).read_bytes() == data for name, data in before.items())


def test_cli_receipt_pair_and_typed_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer
    from typer.testing import CliRunner

    from market_predictor.commands import investment_research as commands

    app = typer.Typer()
    commands.register_investment_research_commands(app)
    calls = []
    monkeypatch.setattr(commands, "materialize_investment_labels", lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    arguments = ["materialize-investment-labels", "--root", str(tmp_path), "--request", str(tmp_path / "request.json"),
        "--request-sha256", "a" * 64, "--output", str(tmp_path / "output"), "--parent-receipt", str(tmp_path / "receipt.json")]
    runner = CliRunner()
    assert runner.invoke(app, arguments).exit_code == 2
    assert calls == []
    assert runner.invoke(app, arguments[:-2] + ["--parent-receipt-sha256", "b" * 64]).exit_code == 2
    assert calls == []
    result = runner.invoke(app, arguments + ["--parent-receipt-sha256", "b" * 64])
    assert result.exit_code == 0, result.output
    assert calls[0]["parent_receipt"] == SourcePin(path=(tmp_path / "receipt.json").as_posix(), sha256="b" * 64)
    result = runner.invoke(app, arguments[:-2])
    assert result.exit_code == 0, result.output
    assert calls[1]["parent_receipt"] is None


def test_generated_python_cache_does_not_change_export_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = fixture(tmp_path, monkeypatch)
    path = value["package"] / "research/__pycache__/investment_label_reuse.cpython-314.pyc"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"generated cache")
    assert module.verify_parent_checkpoints(**value["kwargs"]).receipt.declared_rows == 2
