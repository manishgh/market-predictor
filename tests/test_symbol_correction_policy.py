"""Small policy-reader tests; no archives, collection or model loads."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.universe.symbol_correction_policy import SymbolCorrectionPolicy, load_symbol_correction_policy

_CONFIG = Path(__file__).resolve().parents[1] / "configs/swing_symbol_corrections.toml"


def test_policy_reader_preserves_the_complete_reviewed_contract(tmp_path: Path) -> None:
    config = tmp_path / "corrections.toml"
    config.write_bytes(_CONFIG.read_bytes())
    policy = load_symbol_correction_policy(tmp_path, Path(config.name), file_sha256(config))
    assert policy.model_dump(mode="json") == tomllib.loads(config.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="frozen"):
        policy.parent_plan = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        policy.corrections[0].ticker = "CHANGED"


@pytest.mark.parametrize("field", ["parent_plan_sha256", "parent_archive_sha256", "document_report_sha256"])
@pytest.mark.parametrize("invalid", ["a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_policy_hash_fields_keep_exact_lowercase_sha256_validation(field: str, invalid: str) -> None:
    payload = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    payload[field] = invalid
    with pytest.raises(ValidationError):
        SymbolCorrectionPolicy.model_validate(payload)


@pytest.mark.parametrize("field,invalid", [
    ("security_id", "cik:123"), ("ticker", "sats"), ("provider_symbol", "fi"),
    ("parent_unit_id", "other-daily-" + "a" * 24), ("start_date", "not-a-date"),
    ("end_date", "not-a-date"), ("document_ids", ["one"]),
    ("record_locators", ["one"]), ("interpretation", "short"), ("extra", "forbidden"),
])
def test_correction_fields_keep_existing_validation(field: str, invalid: object) -> None:
    payload = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["corrections"][0][field] = invalid
    with pytest.raises(ValidationError):
        SymbolCorrectionPolicy.model_validate(payload)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_policy_requires_exactly_two_corrections(count: int) -> None:
    payload = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["corrections"] = [payload["corrections"][0]] * count
    with pytest.raises(ValidationError):
        SymbolCorrectionPolicy.model_validate(payload)


@pytest.mark.parametrize("fault", ["pin", "size", "extra", "schema"])
def test_policy_reader_fails_closed(tmp_path: Path, fault: str) -> None:
    config = tmp_path / "corrections.toml"
    payload = _CONFIG.read_bytes()
    if fault == "size":
        payload = b" " * (1024**2 + 1)
    elif fault == "extra":
        payload = b'extra = "forbidden"\n' + payload
    elif fault == "schema":
        payload = payload.replace(b"market_predictor.swing_symbol_correction_policy", b"unsupported")
    config.write_bytes(payload)
    expected = "f" * 64 if fault == "pin" else file_sha256(config)
    error = DataReadinessError if fault in {"pin", "size"} else ValidationError
    with pytest.raises(error):
        load_symbol_correction_policy(tmp_path, config, expected)


def test_shared_inside_preserves_directory_and_prospective_path_semantics(tmp_path: Path) -> None:
    directory = tmp_path / "archive"
    directory.mkdir()
    assert inside(tmp_path, "archive") == directory.resolve()
    assert inside(tmp_path, directory) == directory.resolve()
    assert inside(tmp_path, "future/output") == (tmp_path / "future/output").resolve()
    for invalid in (tmp_path, Path("."), Path(".."), tmp_path.parent / "escape"):
        with pytest.raises(DataReadinessError, match="escapes repository or targets root"):
            inside(tmp_path, invalid)


def test_policy_reader_rejects_paths_outside_authority(tmp_path: Path) -> None:
    config = tmp_path / "corrections.toml"
    config.write_bytes(_CONFIG.read_bytes())
    root = tmp_path / "authority"
    root.mkdir()
    with pytest.raises(DataReadinessError, match="escapes authority root"):
        load_symbol_correction_policy(root, config, file_sha256(config))
