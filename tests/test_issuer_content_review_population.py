from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import issuer_content_review_population as population
from market_predictor.research.issuer_content_review_sources import AliasProof, ReviewSourceRecord


def _record(*, revision: str = "2020-06-01T20:00:00Z", source_id: str = "story") -> ReviewSourceRecord:
    payload = {"id": source_id, "created_at": "2020-06-01T20:00:00Z", "updated_at": revision,
               "headline": "Apex reports Q2 2020 earnings of $1 per share.", "content": None, "summary": None}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()
    return ReviewSourceRecord("alpaca", source_id, digest, "cik:0000000001", "APX", pd.Timestamp(payload["created_at"]),
        pd.Timestamp(revision), pd.Timestamp("2026-09-01T00:00:00Z"), payload, "headline", "/saved/page.json#0",
        {"query_identity_resolution": "identity_equal", "available_at_utc": revision, "query_security_id": "cik:0000000001"}, ())


def _proof(*, name: str = "Apex", date: str = "2020-01-01T00:00:00Z") -> AliasProof:
    return AliasProof("0000000001", (name,), pd.Timestamp(date), "a" * 64, "b" * 64, "index#companyInfo")


def test_alias_selection_never_uses_later_company_name(tmp_path: Path) -> None:
    with population._connect(tmp_path / "test.sqlite") as connection:
        for proof in (_proof(), _proof(name="Later Name", date="2021-01-01T00:00:00Z")):
            population._store_alias(connection, "cik:0000000001", proof)
        aliases = population._Aliases(connection)
        assert aliases.at("cik:0000000001", pd.Timestamp("2019-01-01T00:00:00Z")) is None
        result = population._content_row(_record(), aliases, {})
        assert result[2]["alias_proof"]["aliases"] == ["Apex"]
        assert result[4][0]["event_family"] == "earnings"


def test_readable_missing_alias_is_not_silently_dropped_from_review(tmp_path: Path) -> None:
    with population._connect(tmp_path / "test.sqlite") as connection:
        _, _, row, text, candidates = population._content_row(_record(), population._Aliases(connection), {})
    assert text == "Apex reports Q2 2020 earnings of $1 per share."
    assert candidates == []
    assert "no_unambiguous_issuer_alias_available_by_source_version" in row["unavailable_reasons"]
    assert "event_available_at_utc" not in row


def test_future_revision_is_metadata_only_and_never_blind_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population, "_guard", lambda: None)
    with population._connect(tmp_path / "test.sqlite") as connection:
        population._store_alias(connection, "cik:0000000001", _proof())
        aliases = population._Aliases(connection)
        for record in (_record(), _record(revision="2026-08-01T00:00:00Z")):
            cluster, version, row, text, candidates = population._content_row(record, aliases, {})
            connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
                (version, cluster, record.security_id, "alpaca", 2020, population._json(row), text, population._json(candidates)))
        assert text is None
        assert "version_after_initial_fit_cutoff" in row["unavailable_reasons"]
        clusters = population._clusters(connection)
        assert len(clusters) == 1
        population._export_review(connection, tmp_path, clusters)
        for exported in (tmp_path / "blind").glob("*.json"):
            value = json.loads(exported.read_text(encoding="utf-8"))
            assert len(value["versions"]) == 1
            assert value["versions"][0]["version_available_at_utc"].startswith("2020-")
            assert "candidate_families" not in value
            assert "role" not in value


def _fake_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, Path]:
    root = tmp_path
    (root / "data/research").mkdir(parents=True)
    (root / "configs").mkdir()
    pin_path = root / "configs/source.json"
    pin_path.write_text("{}", encoding="utf-8")
    pin = {"path": "configs/source.json", "sha256": file_sha256(pin_path)}
    settings = {key: pin for key in population._CONFIG_KEYS if key != "schema"} | {"schema": population.CONFIG_SCHEMA}
    config = root / "configs/review.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(population, "_guard", lambda: None)
    monkeypatch.setattr(population, "_identity_clocks", lambda *args: {})
    monkeypatch.setattr(population, "read_early_alias_proofs", lambda **kwargs: (_proof(),))
    monkeypatch.setattr(population, "iter_sec_review_sources", lambda **kwargs: iter(()))
    monkeypatch.setattr(population, "iter_alpaca_review_sources", lambda **kwargs: iter((_record(),)))
    # Production hashes package sources outside this synthetic root. Keep the
    # synthetic fixture honest by rechecking its real input pins only here.
    recheck = population.recheck_source_pins
    monkeypatch.setattr(population, "recheck_source_pins",
                        lambda selected_root, pins: recheck(selected_root, {k: v for k, v in pins.items() if not k.startswith("src/")}))
    return config, file_sha256(config), root / "data/research/review"


def test_publication_is_nonadmitting_blind_and_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, digest, output = _fake_sources(tmp_path, monkeypatch)
    result = population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)
    assert result["status"] == "complete_review_population_only"
    assert result["totals"]["source_versions"] == 1
    assert not result["training_eligible"] and not result["serving_eligible"]
    assert "population.sqlite" in result["artifacts"]
    with pytest.raises(DataReadinessError, match="immutable"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)


def test_input_changed_during_job_prevents_complete_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, digest, output = _fake_sources(tmp_path, monkeypatch)

    def mutated(**kwargs: Any) -> Any:
        yield _record()
        config.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(population, "iter_alpaca_review_sources", mutated)
    with pytest.raises(DataReadinessError, match="changed"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)
    assert not (output / "_manifest.json").exists()


def test_resume_requires_exact_checkpoint_and_does_not_duplicate_source_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, digest, output = _fake_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(population, "COMMIT_ROWS", 1)

    def interrupted(**kwargs: Any) -> Any:
        yield _record()
        raise RuntimeError("interrupted after checkpoint")

    monkeypatch.setattr(population, "iter_alpaca_review_sources", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)
    checkpoint = file_sha256(output / "_checkpoint.json")
    monkeypatch.setattr(population, "iter_alpaca_review_sources", lambda **kwargs: iter((_record(),)))
    with pytest.raises(DataReadinessError, match="checkpoint"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output,
                                                    resume_checkpoint_sha256="0" * 64)
    result = population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output,
                                                          resume_checkpoint_sha256=checkpoint)
    assert result["totals"]["source_occurrences"]["alpaca"] == 1
    assert result["totals"]["source_versions"] == 1


def test_same_issuer_conflicting_names_at_one_clock_are_unavailable(tmp_path: Path) -> None:
    with population._connect(tmp_path / "test.sqlite") as connection:
        population._store_alias(connection, "cik:0000000001", _proof())
        population._store_alias(connection, "cik:0000000001", _proof(name="Competing"))
        assert population._Aliases(connection).at("cik:0000000001", _record().version_available_at_utc) is None


def test_checkpoint_pointer_failure_restores_last_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, digest, output = _fake_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(population, "COMMIT_ROWS", 1)
    records = (_record(), _record(source_id="second"))
    monkeypatch.setattr(population, "iter_alpaca_review_sources", lambda **kwargs: iter(records))
    original = population._atomic

    def fail_after_database_commit(path: Path, value: dict[str, Any]) -> None:
        if value["phase"] == "alpaca" and value["rows_completed"] == 2:
            raise RuntimeError("checkpoint pointer interrupted")
        original(path, value)

    monkeypatch.setattr(population, "_atomic", fail_after_database_commit)
    with pytest.raises(RuntimeError, match="pointer interrupted"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)
    checkpoint = file_sha256(output / "_checkpoint.json")
    value = json.loads((output / "_checkpoint.json").read_text(encoding="utf-8"))
    assert value["rows_completed"] == 1
    assert file_sha256(output / "population.sqlite") != value["database_sha256"]
    monkeypatch.setattr(population, "_atomic", original)
    result = population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output,
                                                          resume_checkpoint_sha256=checkpoint)
    assert result["totals"]["source_occurrences"]["alpaca"] == 2
    assert result["totals"]["source_versions"] == 2


def test_disk_limit_preserves_previous_resume_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, digest, output = _fake_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(population, "COMMIT_ROWS", 1)
    disk_usage = population.shutil.disk_usage
    monkeypatch.setattr(population.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=0) if (output / "_checkpoint.json").exists() else disk_usage(path))
    with pytest.raises(DataReadinessError, match="insufficient disk"):
        population.publish_content_review_population(root=tmp_path, config=config, config_sha256=digest, output=output)
    checkpoint = json.loads((output / "_checkpoint.json").read_text(encoding="utf-8"))
    assert file_sha256(output / checkpoint["database_snapshot"]) == checkpoint["database_sha256"]
    assert not (output / "_manifest.json").exists()


def test_unmapped_identity_keeps_original_query_disposition(tmp_path: Path) -> None:
    record = replace(_record(), security_id=None, unavailable_reasons=("missing_cohort_identity",))
    with population._connect(tmp_path / "test.sqlite") as connection:
        _, _, row, text, candidates = population._content_row(record, population._Aliases(connection), {})
    assert text and not candidates
    assert row["security_id"] is None
    assert "missing_cohort_identity" in row["unavailable_reasons"]


def test_prepare_review_cli_passes_exact_pins_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.commands import issuer_content_inventory as commands
    from market_predictor.research_cli import app

    calls: list[dict[str, Any]] = []

    def publish(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete_review_population_only", "totals": {}, "manifest_sha256": "a" * 64,
                "training_eligible": False, "serving_eligible": False}

    monkeypatch.setattr(commands, "publish_content_review_population", publish)
    result = CliRunner().invoke(app, ["prepare-issuer-content-review", "--root", str(tmp_path),
        "--config", "configs/review.json", "--config-sha256", "b" * 64,
        "--output", "data/research/review", "--resume-checkpoint-sha256", "c" * 64])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "config": Path("configs/review.json"), "config_sha256": "b" * 64,
                      "output": Path("data/research/review"), "resume_checkpoint_sha256": "c" * 64}]
