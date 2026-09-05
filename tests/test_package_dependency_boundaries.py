from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "market_predictor"
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
TEST_ROOT = REPOSITORY_ROOT / "tests"
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
PRODUCTION_PACKAGES = (
    "core",
    "sources",
    "evidence",
    "universe",
    "catalysts",
    "modeling",
    "swing",
    "intraday",
    "governance",
    "serving",
)
FORBIDDEN_DEPENDENCIES = ("market_predictor.research", "market_predictor.commands")
MODELING_FORBIDDEN_DEPENDENCIES = (
    "market_predictor.swing",
    "market_predictor.intraday",
)
UNIVERSE_ALLOWED_DEPENDENCIES = (
    "market_predictor.core",
    "market_predictor.evidence",
    "market_predictor.sources",
    "market_predictor.universe",
    "market_predictor.canonical",
    "market_predictor.locking",
    "market_predictor.resources",
)
SOURCES_ALLOWED_DEPENDENCIES = (
    "market_predictor.config",
    "market_predictor.core",
    "market_predictor.evidence",
    "market_predictor.locking",
    "market_predictor.resources",
    "market_predictor.schemas",
    "market_predictor.sources",
)
CATALYSTS_ALLOWED_DEPENDENCIES = (
    "market_predictor.canonical",
    "market_predictor.catalysts",
    "market_predictor.core",
    "market_predictor.evidence",
    "market_predictor.resources",
    "market_predictor.sources",
    "market_predictor.universe",
)
REMOVED_PRODUCTION_MODULES = (
    "market_predictor.edge_rebuild.catalyst_authority",
    "market_predictor.edge_rebuild.corpus_integrity",
    "market_predictor.edge_rebuild.cross_sectional",
    "market_predictor.edge_rebuild.global_event_authority",
    "market_predictor.edge_rebuild.global_event_collection",
    "market_predictor.edge_rebuild.history_collection",
    "market_predictor.edge_rebuild.history_materialization",
    "market_predictor.edge_rebuild.history_contracts",
    "market_predictor.edge_rebuild.issuer_event_family_authority",
    "market_predictor.edge_rebuild.issuer_event_precision_audit",
    "market_predictor.edge_rebuild.labeling",
    "market_predictor.edge_rebuild.pipeline",
    "market_predictor.edge_rebuild.sec_filing_authority",
    "market_predictor.edge_rebuild.sec_filing_collection",
    "market_predictor.edge_rebuild.sec_identity_authority",
    "market_predictor.edge_rebuild.selected_session_history",
    "market_predictor.edge_rebuild.sp500_memberships",
    "market_predictor.edge_rebuild.sp500_observed_memberships",
    "market_predictor.edge_rebuild.sp500_transitions",
    "market_predictor.edge_rebuild.strategy_contract",
    "market_predictor.edge_rebuild.swing_artifact_contracts",
    "market_predictor.edge_rebuild.technical_relationships",
    "market_predictor.edge_rebuild.universe_identity",
    "market_predictor.edge_rebuild.volume_bars",
    "market_predictor.swing.news_history",
    "market_predictor.swing.news_history_audit",
    "market_predictor.swing.event_attribution",
    "market_predictor.swing.event_attribution_history",
    "market_predictor.swing.event_families",
    "market_predictor.swing.event_relevance",
    "market_predictor.symbols",
)
REMOVED_EDGE_REBUILD_FILES = (
    "catalyst_authority.py",
    "corpus_integrity.py",
    "cross_sectional.py",
    "global_event_authority.py",
    "global_event_collection.py",
    "history_collection.py",
    "history_materialization.py",
    "history_contracts.py",
    "issuer_event_family_authority.py",
    "issuer_event_precision_audit.py",
    "labeling.py",
    "pipeline.py",
    "sec_filing_authority.py",
    "sec_filing_collection.py",
    "sec_identity_authority.py",
    "selected_session_history.py",
    "sp500_memberships.py",
    "sp500_observed_memberships.py",
    "sp500_transitions.py",
    "strategy_contract.py",
    "swing_artifact_contracts.py",
    "technical_relationships.py",
    "universe_identity.py",
    "volume_bars.py",
)
REMOVED_MIGRATED_FILES = (
    "symbols.py",
    "swing/news_history.py",
    "swing/news_history_audit.py",
    "swing/event_attribution.py",
    "swing/event_attribution_history.py",
    "swing/event_families.py",
    "swing/event_relevance.py",
    "swing/contracts.py",
    "swing/labels.py",
)
REMOVED_ACTIVE_SYMBOLS = (
    "GLOBAL_EVENT_QUERY_POLICY_V1",
    "GdeltCollectionRequest",
    "GdeltFetchResult",
    "GdeltFetcher",
    "fetch_gdelt_doc_api",
    "validate_gdelt_collection_request",
)


def test_production_packages_do_not_depend_on_research_or_command_adapters() -> None:
    violations: list[str] = []
    for package_name in PRODUCTION_PACKAGES:
        package = PACKAGE_ROOT / package_name
        if not package.exists():
            continue
        for path in package.rglob("*.py"):
            violations.extend(_forbidden_imports(path))

    assert not violations, "Production dependency violations:\n" + "\n".join(sorted(violations))


def test_chronology_named_v3_package_is_absent() -> None:
    assert not (PACKAGE_ROOT / "v3").exists()


def test_production_tree_has_no_module_package_name_collisions() -> None:
    assert not _module_package_collisions(PACKAGE_ROOT)


def test_modeling_package_is_horizon_neutral() -> None:
    violations: list[str] = []
    for path in (PACKAGE_ROOT / "modeling").rglob("*.py"):
        for node, imported_name in _module_imports(path):
            if _matches_any_dependency(
                imported_name,
                MODELING_FORBIDDEN_DEPENDENCIES,
            ):
                relative_path = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(
                    f"{relative_path}:{node.lineno}: {imported_name}"
                )

    assert not violations, "Modeling horizon violations:\n" + "\n".join(
        sorted(violations)
    )


@pytest.mark.parametrize(
    ("statement", "package_name"),
    (
        ("import market_predictor.swing", "market_predictor.modeling"),
        (
            "import market_predictor.intraday.features as features",
            "market_predictor.modeling",
        ),
        ("from market_predictor import swing", "market_predictor.modeling"),
        (
            "from market_predictor.intraday import dataset",
            "market_predictor.modeling",
        ),
        ("from .. import swing", "market_predictor.modeling"),
        (
            "from ..intraday.features import labels",
            "market_predictor.modeling",
        ),
    ),
)
def test_modeling_horizon_guard_recognizes_every_import_form(
    statement: str,
    package_name: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node, package_name=package_name)
    )
    assert any(
        _matches_any_dependency(name, MODELING_FORBIDDEN_DEPENDENCIES)
        for name in imported_names
    )


def test_module_package_collision_guard_detects_shadow_module(tmp_path: Path) -> None:
    package = tmp_path / "example"
    package.mkdir()
    (package / "signals.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "signals").mkdir()
    (package / "signals" / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert _module_package_collisions(package) == ("signals.py <-> signals/",)


def test_removed_intraday_shadow_modules_are_absent() -> None:
    assert not (PACKAGE_ROOT / "intraday" / "contracts.py").exists()
    assert not (PACKAGE_ROOT / "intraday" / "evaluation.py").exists()


def test_universe_package_uses_only_approved_dependency_layers() -> None:
    violations: list[str] = []
    for path in (PACKAGE_ROOT / "universe").rglob("*.py"):
        for node, imported_name in _module_imports(path):
            if not imported_name.startswith("market_predictor."):
                continue
            if _matches_any_dependency(imported_name, UNIVERSE_ALLOWED_DEPENDENCIES):
                continue
            relative_path = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{relative_path}:{node.lineno}: {imported_name}")

    assert not violations, "Universe dependency violations:\n" + "\n".join(sorted(violations))


def test_source_package_uses_only_approved_dependency_layers() -> None:
    violations: list[str] = []
    for path in (PACKAGE_ROOT / "sources").rglob("*.py"):
        for node, imported_name in _module_imports(path):
            if not imported_name.startswith("market_predictor."):
                continue
            if _matches_any_dependency(imported_name, SOURCES_ALLOWED_DEPENDENCIES):
                continue
            relative_path = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{relative_path}:{node.lineno}: {imported_name}")

    assert not violations, "Source dependency violations:\n" + "\n".join(sorted(violations))


def test_catalyst_package_uses_only_approved_dependency_layers() -> None:
    violations: list[str] = []
    for path in (PACKAGE_ROOT / "catalysts").rglob("*.py"):
        for node, imported_name in _module_imports(path):
            if not imported_name.startswith("market_predictor."):
                continue
            if _matches_any_dependency(imported_name, CATALYSTS_ALLOWED_DEPENDENCIES):
                continue
            relative_path = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{relative_path}:{node.lineno}: {imported_name}")

    assert not violations, "Catalyst dependency violations:\n" + "\n".join(sorted(violations))


def test_removed_production_modules_have_no_imports_or_files() -> None:
    violations: list[str] = []
    for root in (PACKAGE_ROOT, TEST_ROOT, SCRIPT_ROOT):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            for node, imported_name in _module_imports(path):
                if _matches_any_dependency(imported_name, REMOVED_PRODUCTION_MODULES):
                    relative_path = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative_path}:{node.lineno}: {imported_name}")

    old_package = PACKAGE_ROOT / "edge_rebuild"
    remaining_files = [str(old_package / name) for name in REMOVED_EDGE_REBUILD_FILES if (old_package / name).exists()]
    remaining_files.extend(
        str(PACKAGE_ROOT / relative_path)
        for relative_path in REMOVED_MIGRATED_FILES
        if (PACKAGE_ROOT / relative_path).exists()
    )
    assert not remaining_files, "Removed production files still exist:\n" + "\n".join(remaining_files)
    assert not violations, "Removed production imports remain:\n" + "\n".join(sorted(violations))


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.corpus_integrity",
        "import market_predictor.edge_rebuild.corpus_integrity as integrity",
        "from market_predictor.edge_rebuild import corpus_integrity",
        "from market_predictor.edge_rebuild.corpus_integrity import IntegrityThresholds",
    ),
)
def test_removed_authority_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.sec_filing_collection",
        "import market_predictor.edge_rebuild.sec_filing_collection as collection",
        "from market_predictor.edge_rebuild import sec_filing_collection",
        "from market_predictor.edge_rebuild.sec_filing_collection import SecFilingCollection",
        "import market_predictor.edge_rebuild.sec_filing_authority",
        "import market_predictor.edge_rebuild.sec_filing_authority as authority",
        "from market_predictor.edge_rebuild import sec_filing_authority",
        "from market_predictor.edge_rebuild.sec_filing_authority import SecFilingDecisionAuthority",
    ),
)
def test_removed_sec_filing_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.global_event_collection",
        "from market_predictor.edge_rebuild import global_event_collection",
        "from market_predictor.edge_rebuild.global_event_collection import GdeltGlobalEventCollection",
        "import market_predictor.edge_rebuild.global_event_authority",
        "from market_predictor.edge_rebuild import global_event_authority",
        "from market_predictor.edge_rebuild.global_event_authority import GlobalEventAuthority",
    ),
)
def test_removed_global_event_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.symbols",
        "from market_predictor import symbols",
        "from market_predictor.symbols import canonical_symbol",
        "import market_predictor.swing.news_history",
        "from market_predictor.swing import news_history",
        "from market_predictor.swing.news_history import collect_alpaca_news_history",
        "import market_predictor.swing.news_history_audit",
        "from market_predictor.swing.news_history_audit import audit_alpaca_news_history",
    ),
)
def test_removed_issuer_news_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.swing.event_families",
        "from market_predictor.swing import event_families",
        "from market_predictor.swing.event_families import classify_event_families",
        "import market_predictor.swing.event_relevance",
        "from market_predictor.swing.event_relevance import add_event_relevance",
        "import market_predictor.swing.event_attribution",
        "from market_predictor.swing.event_attribution import build_event_security_relations",
        "import market_predictor.swing.event_attribution_history",
        "from market_predictor.swing.event_attribution_history import load_event_attribution_history",
    ),
)
def test_removed_issuer_event_foundation_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.catalyst_authority",
        "import market_predictor.edge_rebuild.catalyst_authority as authority",
        "from market_predictor.edge_rebuild import catalyst_authority",
        "from market_predictor.edge_rebuild.catalyst_authority import CatalystDecisionAuthority",
    ),
)
def test_removed_catalyst_decision_authority_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.issuer_event_family_authority",
        "import market_predictor.edge_rebuild.issuer_event_family_authority as authority",
        "from market_predictor.edge_rebuild import issuer_event_family_authority",
        "from market_predictor.edge_rebuild.issuer_event_family_authority import IssuerEventFamilyAuthority",
    ),
)
def test_removed_issuer_family_authority_import_guard_recognizes_every_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, REMOVED_PRODUCTION_MODULES) for name in imported_names)


def test_swing_and_intraday_packages_do_not_import_each_other() -> None:
    violations: list[str] = []
    boundaries = (
        (PACKAGE_ROOT / "swing", ("market_predictor.intraday",)),
        (PACKAGE_ROOT / "intraday", ("market_predictor.swing",)),
    )
    for package, forbidden_dependencies in boundaries:
        for path in package.rglob("*.py"):
            for node, imported_name in _module_imports(path):
                if _matches_any_dependency(imported_name, forbidden_dependencies):
                    relative_path = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative_path}:{node.lineno}: {imported_name}")

    assert not violations, "Cross-horizon dependency violations:\n" + "\n".join(sorted(violations))


@pytest.mark.parametrize(
    ("statement", "forbidden_dependency"),
    (
        ("import market_predictor.intraday", "market_predictor.intraday"),
        ("from market_predictor import intraday", "market_predictor.intraday"),
        ("from market_predictor.intraday.datasets import event_preflight", "market_predictor.intraday"),
        ("import market_predictor.swing", "market_predictor.swing"),
        ("from market_predictor import swing", "market_predictor.swing"),
        ("from market_predictor.swing.features import catalyst_decision_authority", "market_predictor.swing"),
    ),
)
def test_cross_horizon_dependency_guard_recognizes_import_forms(
    statement: str,
    forbidden_dependency: str,
) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, (forbidden_dependency,)) for name in imported_names)


def test_rule_variant_helper_has_one_semantic_owner() -> None:
    expected_owner = PACKAGE_ROOT / "catalysts" / "issuer_events" / "classification.py"
    definitions: list[Path] = []
    forbidden_imports: list[str] = []
    for root in (PACKAGE_ROOT, TEST_ROOT, SCRIPT_ROOT):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "issuer_event_rule_variant"
                for node in ast.walk(tree)
            ):
                definitions.append(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "market_predictor.edge_rebuild.issuer_event_precision_audit":
                    continue
                if any(alias.name == "issuer_event_rule_variant" for alias in node.names):
                    forbidden_imports.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")

    assert definitions == [expected_owner]
    assert not forbidden_imports, "Rule-variant consumers import the old owner:\n" + "\n".join(forbidden_imports)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.issuer_event_precision_audit",
        "import market_predictor.edge_rebuild.issuer_event_precision_audit as precision",
        "from market_predictor.edge_rebuild import issuer_event_precision_audit",
        "from market_predictor.edge_rebuild.issuer_event_precision_audit import IssuerEventPrecisionAudit",
    ),
)
def test_removed_precision_audit_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.strategy_contract",
        "import market_predictor.edge_rebuild.strategy_contract as contract",
        "from market_predictor.edge_rebuild import strategy_contract",
        "from market_predictor.edge_rebuild.strategy_contract import StrategyContract",
    ),
)
def test_removed_strategy_contract_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.pipeline",
        "import market_predictor.edge_rebuild.pipeline as pipeline",
        "from market_predictor.edge_rebuild import pipeline",
        "from market_predictor.edge_rebuild.pipeline import FeaturePipeline",
    ),
)
def test_removed_feature_pipeline_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.history_contracts",
        "import market_predictor.edge_rebuild.history_contracts as contracts",
        "from market_predictor.edge_rebuild import history_contracts",
        "from market_predictor.edge_rebuild.history_contracts import IntradayHistoryConfig",
    ),
)
def test_removed_intraday_history_contract_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.swing_artifact_contracts",
        "import market_predictor.edge_rebuild.swing_artifact_contracts as contracts",
        "from market_predictor.edge_rebuild import swing_artifact_contracts",
        "from market_predictor.edge_rebuild.swing_artifact_contracts import SWING_MATERIALIZATION_AUTHORITY_SCHEMA",
    ),
)
def test_removed_swing_materialization_contract_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.technical_relationships",
        "import market_predictor.edge_rebuild.technical_relationships as relationships",
        "from market_predictor.edge_rebuild import technical_relationships",
        "from market_predictor.edge_rebuild.technical_relationships import TechnicalRelationshipSpec",
    ),
)
def test_removed_swing_technical_relationship_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.cross_sectional",
        "import market_predictor.edge_rebuild.cross_sectional as cross_sectional",
        "from market_predictor.edge_rebuild import cross_sectional",
        "from market_predictor.edge_rebuild.cross_sectional import CrossSectionSpec",
    ),
)
def test_removed_swing_cross_sectional_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.labeling",
        "import market_predictor.edge_rebuild.labeling as labels",
        "from market_predictor.edge_rebuild import labeling",
        "from market_predictor.edge_rebuild.labeling import BarrierSpec",
    ),
)
def test_removed_labeling_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.volume_bars",
        "import market_predictor.edge_rebuild.volume_bars as volume_bars",
        "from market_predictor.edge_rebuild import volume_bars",
        "from market_predictor.edge_rebuild.volume_bars import VolumeBarBuildResult",
    ),
)
def test_removed_volume_bar_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.selected_session_history",
        "import market_predictor.edge_rebuild.selected_session_history as history",
        "from market_predictor.edge_rebuild import selected_session_history",
        "from market_predictor.edge_rebuild.selected_session_history import SelectedSession",
    ),
)
def test_removed_selected_session_history_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.history_materialization",
        "import market_predictor.edge_rebuild.history_materialization as materialization",
        "from market_predictor.edge_rebuild import history_materialization",
        "from market_predictor.edge_rebuild.history_materialization import SessionBounds",
    ),
)
def test_removed_history_materialization_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.edge_rebuild.history_collection",
        "import market_predictor.edge_rebuild.history_collection as collection",
        "from market_predictor.edge_rebuild import history_collection",
        "from market_predictor.edge_rebuild.history_collection import collect_intraday_history",
    ),
)
def test_removed_history_collection_import_guard_recognizes_every_import_form(
    statement: str,
) -> None:
    imported_names = tuple(
        name
        for node in ast.walk(ast.parse(statement))
        for name in _imported_names(node)
    )
    assert any(
        _matches_any_dependency(name, REMOVED_PRODUCTION_MODULES)
        for name in imported_names
    )


def test_issuer_event_precision_governance_is_horizon_neutral() -> None:
    violations: list[str] = []
    package = PACKAGE_ROOT / "governance" / "issuer_event_precision"
    for path in package.rglob("*.py"):
        for node, imported_name in _module_imports(path):
            if imported_name.startswith(("market_predictor.swing", "market_predictor.intraday")):
                relative = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(f"{relative}:{node.lineno}: {imported_name}")
    assert not violations, "Precision governance depends on a trading horizon:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "statement",
    (
        "issuer_event_rule_variant = classification.issuer_event_rule_variant",
        "issuer_event_rule_variant: object = classification.issuer_event_rule_variant",
        "import market_predictor.catalysts.issuer_events.classification as issuer_event_rule_variant",
        "from market_predictor.catalysts.issuer_events.classification import issuer_event_rule_variant",
        "from market_predictor.catalysts.issuer_events.classification import classify_event_families as issuer_event_rule_variant",
        "def issuer_event_rule_variant(row):\n    return row",
    ),
)
def test_rule_variant_owner_guard_detects_module_scope_rebinding(statement: str) -> None:
    assert _top_level_binding_lines(ast.parse(statement), "issuer_event_rule_variant")


def test_removed_global_event_api_names_are_absent_from_python_code() -> None:
    violations: list[str] = []
    for root in (PACKAGE_ROOT, TEST_ROOT, SCRIPT_ROOT):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                identifier = node.id if isinstance(node, ast.Name) else None
                attribute = node.attr if isinstance(node, ast.Attribute) else None
                matched = identifier or attribute
                if matched in REMOVED_ACTIVE_SYMBOLS:
                    relative_path = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative_path}:{node.lineno}: {matched}")

    assert not violations, "Removed global-event API names remain:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "statement",
    (
        "import market_predictor.universe",
        "import market_predictor.universe as universe",
        "from market_predictor import universe",
        "from market_predictor.universe.sp500 import membership_authority",
        "from market_predictor.universe.sp500.membership_authority import publish_sp500_membership_authority",
    ),
)
def test_source_dependency_guard_recognizes_every_universe_import_form(statement: str) -> None:
    tree = ast.parse(statement)
    imported_names = tuple(name for node in ast.walk(tree) for name in _imported_names(node))
    assert any(_matches_any_dependency(name, ("market_predictor.universe",)) for name in imported_names)
    assert not any(_matches_any_dependency(name, SOURCES_ALLOWED_DEPENDENCIES) for name in imported_names)


def _forbidden_imports(path: Path) -> list[str]:
    violations: list[str] = []
    for node, imported_name in _module_imports(path):
        if _matches_any_dependency(imported_name, FORBIDDEN_DEPENDENCIES):
            relative_path = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(f"{relative_path}:{node.lineno}: {imported_name}")
    return violations


def _module_package_collisions(root: Path) -> tuple[str, ...]:
    collisions: list[str] = []
    for module_path in root.rglob("*.py"):
        if module_path.name == "__init__.py":
            continue
        package_path = module_path.with_suffix("")
        if (package_path / "__init__.py").is_file():
            module_relative = module_path.relative_to(root).as_posix()
            package_relative = package_path.relative_to(root).as_posix()
            collisions.append(f"{module_relative} <-> {package_relative}/")
    return tuple(sorted(collisions))


def _top_level_binding_lines(tree: ast.Module, name: str) -> list[int]:
    lines: list[int] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any((alias.asname or alias.name.split(".", 1)[0]) == name for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if any((alias.asname or alias.name) == name for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                lines.append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            lines.append(node.lineno)
    return lines


def _module_imports(path: Path) -> tuple[tuple[ast.AST, str], ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"Cannot inspect invalid Python module {path}: {exc}")

    imports: list[tuple[ast.AST, str]] = []
    package_name = _package_name_for_path(path)
    for node in ast.walk(tree):
        imports.extend(
            (node, imported_name)
            for imported_name in _imported_names(
                node,
                package_name=package_name,
            )
        )
    return tuple(imports)


def _package_name_for_path(path: Path) -> str | None:
    try:
        relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    except ValueError:
        return None
    package_parts = relative.parts[:-1]
    return ".".join(package_parts) if package_parts else None


def _imported_names(
    node: ast.AST,
    *,
    package_name: str | None = None,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            if package_name is None:
                return ()
            package_parts = package_name.split(".")
            retained = len(package_parts) - node.level + 1
            if retained < 1:
                return ()
            module_parts = module.split(".") if module else []
            module = ".".join((*package_parts[:retained], *module_parts))
        if module == "market_predictor":
            return tuple(f"{module}.{alias.name}" for alias in node.names)
        return (module, *(f"{module}.{alias.name}" for alias in node.names))
    return ()


def _matches_any_dependency(imported_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(imported_name == prefix or imported_name.startswith(f"{prefix}.") for prefix in prefixes)
