from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.training import retained_runs as owner
from market_predictor.swing.training import return_artifacts


def write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    policy = json.loads((Path(__file__).parents[1] / "configs/swing_return_training.json").read_text())
    days = [(date(2019, 7, 9) + timedelta(days=i)).isoformat() for i in range(560)]
    folds = [{"number": n + 1, "train": days[:503 + n], "embargo": days[503 + n:513 + n],
              "score": days[513 + n:514 + n]} for n in range(4)]
    source = tmp_path / "opaque.toml"
    source.write_bytes(b"opaque historical mixed strategy, never parse or execute")
    source_pins = {"opaque.toml": hashlib.sha256(source.read_bytes()).hexdigest()}
    for key in ("readiness", "readiness_config"):
        relative_path = f"evidence/{key}.json"
        digest = write(tmp_path / relative_path, {"synthetic": key})
        policy[key] = {"path": relative_path, "sha256": digest}
        source_pins[relative_path] = digest
    request = {"schema": "market_predictor.swing_return_training_request", "policy": policy,
               "feature_names": ["return_5d", "intraday_return"], "holdout_security_ids": ["held-out"],
               "input_rows": 1000, "input_decision_ids_sha256": "a" * 64, "folds": folds,
               "source_files": source_pins,
               **dict.fromkeys(owner.FLAGS, False)}
    directory = tmp_path / "run"
    request_hash = write(directory / "_request.json", request)
    records = {}
    for family, settings in (("regularized_linear_return", policy["linear"]), ("shallow_boosted_return", policy["boosted"])):
        keys = [(f"{scope}/fold-{n}", scope, folds[n - 1]["score"][0]) for n in range(1, 5)
                for scope in ("temporal", "security_transfer")]
        keys.append(("final_refit", "final_refit", folds[-1]["score"][-1]))
        for suffix, scope, cutoff in keys:
            key = f"{family}/{suffix}"
            child = directory / key
            child.mkdir(parents=True)
            files = {}
            for name in (["model.joblib"] if scope == "final_refit" else ["model.joblib", "predictions.parquet"]):
                content = f"not executable: {key}/{name}".encode()
                (child / name).write_bytes(content)
                files[name] = hashlib.sha256(content).hexdigest()
            unit = {"family": family, "scope": scope, "parameters": settings, "feature_names": request["feature_names"],
                    "target": policy["target"], "preprocessing": policy["missingness"], "training_rows": 600,
                    "training_sessions": 503, "scoring_rows": 0 if scope == "final_refit" else 10,
                    "scope_eligible_rows": 0 if scope == "final_refit" else 10, "training_security_ids": ["trained"],
                    "training_decision_ids_sha256": "b" * 64, "training_weights_sha256": "c" * 64,
                    "scoring_decision_ids_sha256": "d" * 64, "fit_cutoff_utc": cutoff + "T22:00:00+00:00",
                    "maximum_training_label_maturity": cutoff + "T20:00:00+00:00"}
            item = {"schema": return_artifacts.SCHEMA, "unit": unit, "files": files, "metrics": {},
                    "request_sha256": request_hash, "serving_eligible": False, "promotion_eligible": False}
            records[key] = {"path": f"run/{key}", "manifest_sha256": write(child / "_manifest.json", item), "metrics": {}}
    checkpoint = {"request_sha256": request_hash, "units": records}
    manifest = {"schema": "market_predictor.swing_return_training", "status": "complete_research_only",
                "specifications": 2, "fold_scope_fits": 16, "final_models": 2, **checkpoint, **dict.fromkeys(owner.FLAGS, False)}
    return tmp_path, {"request_sha256": request_hash, "checkpoint_sha256": write(directory / "_checkpoint.json", checkpoint),
                     "manifest_sha256": write(directory / "_manifest.json", manifest)}


def verify(run: tuple[Path, dict[str, str]]) -> dict[str, Any]:
    root, pins = run
    return owner.verify_completed_run_integrity(root=root, directory=Path("run"), **pins)


def test_complete_integrity_only_without_deserializing(run: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch) -> None:
    deserialize = Mock(side_effect=AssertionError("must not deserialize"))
    monkeypatch.setattr(return_artifacts.joblib, "load", deserialize)
    result = verify(run)
    assert result["scope"] == "historical_integrity_only"
    assert result["verified_units"] == 18
    assert result["source_integrity"] == "matching"
    assert result["current_replay"] == result["numerical_equivalence"] == result["causality"] == "unverified"
    assert result["serving_eligible"] is result["promotion_eligible"] is False
    deserialize.assert_not_called()


@pytest.mark.parametrize("change", ["hash_mismatch", "missing", "unreadable"])
def test_source_discrepancies_are_not_replay_admission(run: tuple[Path, dict[str, str]], change: str,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    source = run[0] / "opaque.toml"
    if change == "missing":
        source.unlink()
    elif change == "hash_mismatch":
        source.write_bytes(b"changed")
    else:
        original = owner._bounded_hash

        def denied(path: Path, limit: int) -> str:
            if path == source:
                raise PermissionError("denied")
            return original(path, limit)

        monkeypatch.setattr(owner, "_bounded_hash", denied)
    result = verify(run)
    assert result["source_integrity"] == "discrepancies"
    assert result["source_files"][0]["status"] == change
    assert result["current_replay"] == "unverified"


@pytest.mark.parametrize("filename", ["_manifest.json", "_request.json", "_checkpoint.json"])
def test_independent_root_pins_required(run: tuple[Path, dict[str, str]], filename: str) -> None:
    with (run[0] / "run" / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DataReadinessError, match="external pin"):
        verify(run)


@pytest.mark.parametrize("field", ["readiness", "readiness_config"])
@pytest.mark.parametrize("poison", ["missing", "different_hash"])
def test_policy_source_pins_must_be_in_the_checked_inventory(
    run: tuple[Path, dict[str, str]], field: str, poison: str,
) -> None:
    request = json.loads((run[0] / "run/_request.json").read_text())
    path = request["policy"][field]["path"]
    if poison == "missing":
        del request["source_files"][path]
    else:
        request["source_files"][path] = "f" * 64
    with pytest.raises(DataReadinessError, match="policy source pin"):
        owner._request(request)


@pytest.mark.parametrize("poison", ["missing_unit", "extra_unit", "checkpoint", "status", "promotion", "path", "count"])
def test_root_inventory_poison(run: tuple[Path, dict[str, str]], poison: str) -> None:
    root, pins = run
    path = root / "run/_manifest.json"
    manifest = json.loads(path.read_text())
    checkpoint = json.loads((root / "run/_checkpoint.json").read_text())
    key = next(iter(manifest["units"]))
    if poison == "missing_unit":
        del manifest["units"][key]
    elif poison == "extra_unit":
        manifest["units"]["other"] = manifest["units"][key]
    elif poison == "status":
        manifest["status"] = "running"
    elif poison == "promotion":
        manifest["promotion_eligible"] = True
    elif poison == "path":
        manifest["units"][key]["path"] = "../escape"
    elif poison == "count":
        manifest["final_models"] = True
    if poison != "checkpoint":
        checkpoint["units"] = manifest["units"]
    else:
        checkpoint["units"] = {}
    pins["manifest_sha256"] = write(path, manifest)
    pins["checkpoint_sha256"] = write(root / "run/_checkpoint.json", checkpoint)
    with pytest.raises(DataReadinessError):
        verify(run)


@pytest.mark.parametrize("poison", ["family", "features", "parameters", "scope", "holdout", "maturity", "cutoff", "files"])
def test_unit_identity_poison(run: tuple[Path, dict[str, str]], poison: str) -> None:
    root, pins = run
    key = "regularized_linear_return/security_transfer/fold-1"
    path = root / "run" / key / "_manifest.json"
    item = json.loads(path.read_text())
    unit = item["unit"]
    if poison == "family":
        unit["family"] = "shallow_boosted_return"
    elif poison == "features":
        unit["feature_names"].reverse()
    elif poison == "parameters":
        unit["parameters"]["alpha"] = 2.0
    elif poison == "scope":
        unit["scope"] = "temporal"
    elif poison == "holdout":
        unit["training_security_ids"] = ["held-out"]
    elif poison == "maturity":
        unit["maximum_training_label_maturity"] = unit["fit_cutoff_utc"]
    elif poison == "cutoff":
        unit["fit_cutoff_utc"] = "2026-01-01T22:00:00+00:00"
    else:
        item["files"]["../escape"] = "a" * 64
    digest = write(path, item)
    for filename, pin in (("_manifest.json", "manifest_sha256"), ("_checkpoint.json", "checkpoint_sha256")):
        metadata = json.loads((root / "run" / filename).read_text())
        metadata["units"][key]["manifest_sha256"] = digest
        pins[pin] = write(root / "run" / filename, metadata)
    with pytest.raises(DataReadinessError):
        verify(run)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'\xff'])
def test_strict_json(run: tuple[Path, dict[str, str]], payload: bytes) -> None:
    (run[0] / "run/_manifest.json").write_bytes(payload)
    run[1]["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(DataReadinessError):
        verify(run)


def test_payload_tampering_and_bounds(run: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch) -> None:
    path = run[0] / "run/regularized_linear_return/final_refit/model.joblib"
    path.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError, match="payload hash"):
        verify(run)
    monkeypatch.setattr(owner, "MAX_JSON_BYTES", 4)
    with pytest.raises(DataReadinessError, match="exceeds bound"):
        verify(run)


@pytest.mark.parametrize("path", ["../run", "/run", "C:/run", "run/../run", "run\\child", "run/."])
def test_path_containment(run: tuple[Path, dict[str, str]], path: str) -> None:
    with pytest.raises(DataReadinessError):
        owner._path(run[0], path)


def test_cli_verifies_tiny_run_without_mutation(run: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.commands import swing_return_training as command

    app = typer.Typer()
    command.register_swing_return_training_command(app)
    forbidden = Mock(side_effect=AssertionError("training or deserialization called"))
    monkeypatch.setattr(command, "train_swing_returns", forbidden)
    monkeypatch.setattr(return_artifacts.joblib, "load", forbidden)
    root, pins = run
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    args = ["verify-retained-swing-run", "--root", str(root), "--directory", "run"]
    for name, digest in pins.items():
        args.extend(["--" + name.replace("_", "-"), digest])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["scope"] == "historical_integrity_only"
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    forbidden.assert_not_called()


def test_missing_payload_fails(run: tuple[Path, dict[str, str]]) -> None:
    (run[0] / "run/shallow_boosted_return/final_refit/model.joblib").unlink()
    with pytest.raises(DataReadinessError):
        verify(run)


@pytest.mark.parametrize("bound", ["MAX_PAYLOAD_BYTES", "MAX_SOURCE_BYTES", "MAX_SOURCE_TOTAL_BYTES", "MAX_SOURCE_FILES"])
def test_io_bounds(run: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch, bound: str) -> None:
    monkeypatch.setattr(owner, bound, 0)
    with pytest.raises(DataReadinessError):
        verify(run)
