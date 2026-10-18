"""Consume a completed source-link index without reopening producer raw inputs.

The caller holds its heavy-job lease. Original raw hashes remain immutable
producer provenance; the actual SQLite/config/code files are checked at both
ends of this context. Query, revision, clock and proof semantics are inherited
unchanged from the canonical index reader.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.research import news_runtime_memory as runtime
from market_predictor.research import news_source_links as original
from market_predictor.swing.datasets.return_relationship_integrity import pins, read_object

IMPLEMENTATION_PATHS = (
    "research/news_source_index_inputs.py", "swing/datasets/return_relationship_integrity.py",
    "catalysts/issuer_events/news_query_scope.py", "core/json_integrity.py", "evidence/io.py",
)


@dataclass
class ConsumedNewsSourceLinkReader(original.NewsSourceLinkReader):
    """Metadata-only extension: all SQLite reader methods remain inherited."""

    producer_source_provenance: Mapping[str, str]
    consumption_scope: str = "saved_source_link_index"
    raw_sources_rechecked: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): runtime.file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _inventory(directory: Path) -> None:
    expected = {"_manifest.json", "_request.json", "index.sqlite"}
    _require(not directory.name.endswith(".partial")
             and {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} == expected,
             "source-index consumption requires the exact completed file inventory")


@contextmanager
def open_news_source_link_index(*, root: Path, index: SourcePin) -> Iterator[ConsumedNewsSourceLinkReader]:
    """Open the original read-only SQLite reader under the caller's existing lease."""
    root = root.resolve()
    runtime.guard("saved source-link index input")
    files: dict[str, str] = {}
    path = runtime.pin_file(root, index.path, index.sha256, files)
    _require(path.name == "_manifest.json", "source-index consumption requires a manifest pin")
    _inventory(path.parent)
    manifest = read_object(path, index.sha256)
    _require(manifest.get("schema") == original.SCHEMA and manifest.get("status") == "complete_source_links_only"
             and manifest.get("coverage") == "unknown"
             and all(manifest.get(key) is value for key, value in original.CLOSED.items()),
             "source-index publication scope or admission flags differ")
    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict) and set(artifacts) == {"index.sqlite", "_request.json"}
             and artifacts["_request.json"] == manifest["request_sha256"], "source-index artifact inventory differs")
    for name, digest in artifacts.items():
        runtime.pin_file(root, inside(path.parent, name).relative_to(root).as_posix(), digest, files)
    request = read_object(path.parent / "_request.json", manifest["request_sha256"])
    _require(request.get("schema") == original.SCHEMA + "_request"
             and all(request.get(key) is value for key, value in original.CLOSED.items()),
             "source-index request scope or admission flags differ")
    config = SourcePin.model_validate(request["config"])
    config_path = runtime.pin_file(root, config.path, config.sha256, files)
    config_body = read_object(config_path, config.sha256)
    policy = original.NewsSourceLinkPolicy.model_validate_json(original._json(request["policy"]))
    _require(original.NewsSourceLinkPolicy.model_validate_json(original._json(config_body)) == policy
             and len(policy.archives) == 3 and {archive.name for archive in policy.archives} == {"early", "later", "corrected"},
             "source-index current config and archived request policy differ")
    provenance = manifest["source_files"]
    _require(isinstance(provenance, dict), "source-index producer provenance must be a pin mapping")
    # Validate the original map's identities without reading its raw payloads.
    canonical_provenance = pins(root, provenance)
    for pin in (config, policy.corpus_manifest, policy.population_manifest, policy.population_database, policy.identity_manifest,
                *(item for archive in policy.archives for item in (archive.collection_manifest, archive.attribution_manifest))):
        key = inside(root, pin.path).relative_to(root).as_posix()
        _require(canonical_provenance.get(key) == pin.sha256, "source-index policy lacks its original producer source pin")
    original_code = original._implementation(root)
    _require(request["implementation_files"] == manifest["implementation_files"] == original_code,
             "source-index original reader or parser implementation differs")
    implementations = pins(root, original_code, _implementation(root))
    runtime.recheck_files(root, implementations)
    with closing(original._connect(path.parent / "index.sqlite", readonly=True)) as db:
        try:
            yield ConsumedNewsSourceLinkReader(
                MappingProxyType(dict(files)), MappingProxyType(dict(implementations)), db,
                MappingProxyType(dict(provenance)),
            )
        finally:
            runtime.recheck_files(root, files)
            runtime.recheck_files(root, implementations)
            _inventory(path.parent)
