"""Keep implementation iterations out of internal contract identities."""

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "market_predictor"
PUBLIC_IDENTITIES = {
    "market_predictor.prediction.v1",
    "alpaca.news_http_receipt.v1",
    "alpaca.news_collection_owner.v1",
    "alpaca.news_collection_plan.v1",
    "alpaca.news_collection_intent.v1",
    "alpaca.news_collection_result.v1",
}
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")
GENERATION = re.compile(r"(?:[._]v\d+)(?:[._]|$)|\bml_v\d+", re.IGNORECASE)


def test_internal_identities_and_names_have_no_generation_suffix() -> None:
    violations = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                # Provider URLs and retained raw artifact paths are not internal IDs.
                if TOKEN.fullmatch(value) and GENERATION.search(value) and value not in PUBLIC_IDENTITIES:
                    violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}: {value}")
            elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if re.search(r"(?:V\d+|_v\d+)$", node.name):
                    violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}: {node.name}")
        assert not re.search(r"(?:^|[_/])v\d+(?:[_.]|$)", path.relative_to(SOURCE).as_posix(), re.IGNORECASE)
    assert not violations, "\n".join(violations)
