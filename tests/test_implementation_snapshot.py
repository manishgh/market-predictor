"""Exact-byte provenance, immutable resume and fail-closed snapshot tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence import implementation_snapshot as snapshot

_SOURCE = "src/market_predictor/publication.py"
_BYTES = b"# coding: latin-1\r\n# \xe9\r\nraise RuntimeError('never execute')\r\n"


def _fixture(root: Path) -> tuple[dict[str, str], Path]:
    source = root / _SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes(_BYTES)
    return {_SOURCE: hashlib.sha256(_BYTES).hexdigest()}, root / "data/evidence/snapshot"


def _verify(root: Path, receipt: dict[str, str], files: dict[str, str]) -> dict[str, str]:
    return snapshot.verify_implementation_snapshot(
        root, Path(receipt["manifest_path"]), receipt["manifest_sha256"], files,
    )


def _blob(root: Path, receipt: dict[str, str], files: dict[str, str]) -> Path:
    return root / next(name for name in _verify(root, receipt, files) if name.endswith(".bin"))


def test_exact_bytes_repeat_and_migrated_source(tmp_path: Path) -> None:
    files, output = _fixture(tmp_path)
    other = "src/market_predictor/nested/other.py"
    (tmp_path / other).parent.mkdir()
    (tmp_path / other).write_bytes(_BYTES)
    files[other] = files[_SOURCE]
    first = snapshot.create_implementation_snapshot(tmp_path, files, output)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.rglob("*") if path.is_file()}
    assert snapshot.create_implementation_snapshot(tmp_path, dict(reversed(list(files.items()))), output) == first
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    archived = _verify(tmp_path, first, files)
    assert len(archived) == 2
    for name, digest in archived.items():
        if name == first["manifest_path"]:
            assert digest == first["manifest_sha256"]
            continue
        assert name.endswith(f"objects/{digest}.bin")
        assert (tmp_path / name).read_bytes() == _BYTES
    assert not list(output.rglob("*.py"))
    (tmp_path / _SOURCE).write_bytes(b"# migrated\n")
    (tmp_path / other).unlink()
    assert _verify(tmp_path, first, files) == archived
    with pytest.raises(DataReadinessError, match="missing|SHA256 mismatch"):
        snapshot.create_implementation_snapshot(tmp_path, files, output)


@pytest.mark.parametrize("target", ["manifest", "blob"])
def test_tamper_rejected_without_overwrite(tmp_path: Path, target: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    path = tmp_path / receipt["manifest_path"] if target == "manifest" else _blob(tmp_path, receipt, files)
    path.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError):
        _verify(tmp_path, receipt, files)
    with pytest.raises(DataReadinessError):
        snapshot.create_implementation_snapshot(tmp_path, files, output)
    assert path.read_bytes() == b"tampered"


@pytest.mark.parametrize("change", ["extra", "renamed", "hash"])
def test_independent_declaration_required(tmp_path: Path, change: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    changed = dict(files)
    if change == "extra":
        changed["src/market_predictor/extra.py"] = files[_SOURCE]
    elif change == "renamed":
        changed = {"src/market_predictor/renamed.py": files[_SOURCE]}
    else:
        changed[_SOURCE] = "0" * 64
    with pytest.raises(DataReadinessError):
        _verify(tmp_path, receipt, changed)


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65])
def test_bad_hashes(tmp_path: Path, digest: str) -> None:
    files, output = _fixture(tmp_path)
    with pytest.raises(DataReadinessError, match="explicit lowercase SHA256"):
        snapshot.create_implementation_snapshot(tmp_path, {_SOURCE: digest}, output)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    with pytest.raises(DataReadinessError, match="explicit lowercase SHA256"):
        snapshot.verify_implementation_snapshot(tmp_path, Path(receipt["manifest_path"]), digest, files)


@pytest.mark.parametrize("name", [
    "../outside.py", "/src/market_predictor/a.py", "C:/outside.py",
    "src/market_predictor/../a.py", "src/market_predictor//a.py",
    "src/market_predictor/./a.py", "src\\market_predictor\\..\\a.py",
    "src/other/a.py", "src/market_predictor/a.bin", "src/market_predictor/__init__.py",
    "src/market_predictor/nested/__init__.py", "src/market_predictor/a.py:stream",
])
def test_bad_source_paths(tmp_path: Path, name: str) -> None:
    files, output = _fixture(tmp_path)
    with pytest.raises(DataReadinessError, match="source path"):
        snapshot.create_implementation_snapshot(tmp_path, {name: files[_SOURCE]}, output)
    assert not output.exists()


def test_missing_source_and_wrong_source_hash(tmp_path: Path) -> None:
    files, output = _fixture(tmp_path)
    with pytest.raises(DataReadinessError, match="SHA256 mismatch"):
        snapshot.create_implementation_snapshot(tmp_path, {_SOURCE: "0" * 64}, output)
    (tmp_path / _SOURCE).unlink()
    with pytest.raises(DataReadinessError, match="missing"):
        snapshot.create_implementation_snapshot(tmp_path, files, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["manifest", "blob"])
def test_incomplete_snapshot_and_exact_resume(tmp_path: Path, target: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    path = tmp_path / receipt["manifest_path"] if target == "manifest" else _blob(tmp_path, receipt, files)
    path.unlink()
    with pytest.raises(DataReadinessError, match="missing"):
        _verify(tmp_path, receipt, files)
    assert snapshot.create_implementation_snapshot(tmp_path, files, output) == receipt
    assert _verify(tmp_path, receipt, files)


@pytest.mark.parametrize("location", ["outside", "wrong_tree", "evidence_root", "traversal"])
def test_snapshot_root_escape(tmp_path: Path, location: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    bad = {
        "outside": tmp_path.parent / "outside",
        "wrong_tree": tmp_path / "src/market_predictor/archive",
        "evidence_root": tmp_path / "data/evidence",
        "traversal": output / "../escape",
    }[location]
    with pytest.raises(DataReadinessError):
        snapshot.create_implementation_snapshot(tmp_path, files, bad)
    with pytest.raises(DataReadinessError):
        snapshot.verify_implementation_snapshot(tmp_path, bad / "_manifest.json", receipt["manifest_sha256"], files)


@pytest.mark.parametrize("poison", ["extra_key", "duplicate_key", "blob_escape", "noncanonical", "version"])
def test_strict_manifest_even_with_recomputed_pin(tmp_path: Path, poison: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    manifest = tmp_path / receipt["manifest_path"]
    original = manifest.read_bytes()
    value = json.loads(original)
    if poison == "extra_key":
        value["undeclared"] = True
    elif poison == "blob_escape":
        value["files"][_SOURCE]["blob"] = "../../outside.bin"
    elif poison == "version":
        value["schema_version"] = 2
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if poison == "duplicate_key":
        payload = b'{"schema_version":1,' + original[1:]
    elif poison == "noncanonical":
        payload = original + b" "
    manifest.write_bytes(payload)
    with pytest.raises(DataReadinessError, match="manifest"):
        snapshot.verify_implementation_snapshot(tmp_path, manifest, hashlib.sha256(payload).hexdigest(), files)


@pytest.mark.parametrize("bound", ["file", "total", "count", "manifest"])
def test_bounds_on_create_and_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    constant, limit = {
        "file": ("_MAX_FILE_BYTES", len(_BYTES) - 1),
        "total": ("_MAX_TOTAL_BYTES", len(_BYTES) - 1),
        "count": ("_MAX_FILES", 0),
        "manifest": ("_MAX_MANIFEST_BYTES", 10),
    }[bound]
    monkeypatch.setattr(snapshot, constant, limit)
    with pytest.raises(DataReadinessError):
        snapshot.create_implementation_snapshot(tmp_path, files, output)
    with pytest.raises(DataReadinessError):
        _verify(tmp_path, receipt, files)


def test_case_alias_conflict(tmp_path: Path) -> None:
    files, output = _fixture(tmp_path)
    files["src/market_predictor/Publication.py"] = "0" * 64
    with pytest.raises(DataReadinessError, match="conflicting"):
        snapshot.create_implementation_snapshot(tmp_path, files, output)


@pytest.mark.parametrize("target", ["source", "source_parent", "output", "objects", "blob", "manifest", "root"])
def test_symlinks_rejected(tmp_path: Path, target: str) -> None:
    root = tmp_path / "repo"
    files, output = _fixture(root)
    receipt = snapshot.create_implementation_snapshot(root, files, output)
    paths = {
        "source": root / _SOURCE, "source_parent": root / "src/market_predictor",
        "output": output, "objects": output / "objects",
        "blob": _blob(root, receipt, files),
        "manifest": root / receipt["manifest_path"], "root": root,
    }
    path = paths[target]
    moved = tmp_path / "moved"
    is_dir = path.is_dir()
    path.rename(moved)
    try:
        path.symlink_to(moved, target_is_directory=is_dir)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(DataReadinessError, match="symlinks/reparse"):
        snapshot.create_implementation_snapshot(root, files, output)
    if target not in {"source", "source_parent"}:
        with pytest.raises(DataReadinessError, match="symlinks/reparse"):
            _verify(root, receipt, files)


def test_subsets_and_windows_slashes(tmp_path: Path) -> None:
    files, output = _fixture(tmp_path)
    other = "src/market_predictor/other.py"
    (tmp_path / other).write_bytes(b"# other\n")
    files[other] = hashlib.sha256(b"# other\n").hexdigest()
    windows = {name.replace("/", "\\"): digest for name, digest in files.items()}
    receipt = snapshot.create_implementation_snapshot(tmp_path, windows, output)
    assert snapshot.create_implementation_snapshot(tmp_path, files, output) == receipt
    pins = _verify(tmp_path, receipt, {_SOURCE.replace("/", "\\"): files[_SOURCE]})
    assert pins == {
        receipt["manifest_path"]: receipt["manifest_sha256"],
        f"data/evidence/snapshot/objects/{files[_SOURCE]}.bin": files[_SOURCE],
    }
    assert _verify(tmp_path, receipt, {}) == {receipt["manifest_path"]: receipt["manifest_sha256"]}
    with pytest.raises(DataReadinessError):
        snapshot.create_implementation_snapshot(tmp_path, {}, output)
    (output / "objects" / f"{files[other]}.bin").unlink()
    with pytest.raises(DataReadinessError, match="missing"):
        _verify(tmp_path, receipt, {_SOURCE: files[_SOURCE]})


def test_slash_alias_conflicting_hash(tmp_path: Path) -> None:
    files, output = _fixture(tmp_path)
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    files[_SOURCE.replace("/", "\\")] = "0" * 64
    with pytest.raises(DataReadinessError, match="conflicting"):
        snapshot.create_implementation_snapshot(tmp_path, files, output)
    with pytest.raises(DataReadinessError, match="conflicting"):
        _verify(tmp_path, receipt, files)


def test_total_bound_accumulates_across_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files, output = _fixture(tmp_path)
    other = "src/market_predictor/other.py"
    (tmp_path / other).write_bytes(_BYTES)
    files[other] = files[_SOURCE]
    receipt = snapshot.create_implementation_snapshot(tmp_path, files, output)
    monkeypatch.setattr(snapshot, "_MAX_TOTAL_BYTES", 2 * len(_BYTES) - 1)
    with pytest.raises(DataReadinessError, match="byte bound"):
        snapshot.create_implementation_snapshot(tmp_path, files, output)
    with pytest.raises(DataReadinessError, match="byte bound"):
        _verify(tmp_path, receipt, files)
