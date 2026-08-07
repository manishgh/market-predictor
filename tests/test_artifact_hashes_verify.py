"""Every published artifact must still hash to what its manifest claims.

A content hash exists to detect edits made outside the publisher. Recomputing it
after such an edit converts the detector into a rubber stamp, so this test reads
the manifest and re-hashes the bytes rather than trusting any recorded value.

The corruption this test exists to catch actually happened: a one-shot script
rewrote ``memberships.parquet`` in place, degrading ``effective_to_utc`` from
``datetime64[us, UTC]`` to ``str``, and the mismatch went unnoticed until a
manual audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

CANONICAL_ROOT = Path("data/canonical")


def _manifests() -> list[Path]:
    if not CANONICAL_ROOT.exists():
        return []
    return sorted(CANONICAL_ROOT.rglob("*.manifest.json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda p: p.parent.name)
def test_canonical_artifact_matches_its_manifest_hash(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("artifact_sha256")
    if declared is None:
        pytest.skip(f"{manifest_path} declares no artifact_sha256")

    # `foo.parquet.manifest.json` describes `foo.parquet`.
    artifact = manifest_path.with_name(manifest_path.name.removesuffix(".manifest.json"))
    assert artifact.exists(), f"manifest describes a missing artifact: {artifact}"

    observed = _sha256(artifact)
    assert observed == declared, (
        f"{artifact} no longer matches its manifest.\n"
        f"  declared: {declared}\n"
        f"  observed: {observed}\n"
        "The artifact was modified outside the publisher. Restore it or "
        "regenerate it through the collector -- do not patch the manifest."
    )


def test_at_least_one_canonical_manifest_is_checked() -> None:
    """A silently empty parametrization would make the check above vacuous."""

    if not CANONICAL_ROOT.exists():
        pytest.skip("no canonical data present in this checkout")
    assert _manifests(), "canonical data exists but declares no manifests"
