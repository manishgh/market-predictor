from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PLAN = REPOSITORY_ROOT / "docs" / "active_edge_rebuild_plan.md"
ACTIVE_HANDOFF = REPOSITORY_ROOT / "docs" / "reviews" / "active_edge_rebuild_handoff.md"


def test_active_continuity_documents_have_one_current_step() -> None:
    plan = ACTIVE_PLAN.read_text(encoding="utf-8")
    handoff = ACTIVE_HANDOFF.read_text(encoding="utf-8")

    assert plan.count("(`in progress`)") == 1
    assert handoff.count("Exact next checkpoint:") == 1
    assert "### A5.2 -" not in plan


def test_active_continuity_documents_have_no_control_characters() -> None:
    for path in (ACTIVE_PLAN, ACTIVE_HANDOFF):
        text = path.read_text(encoding="utf-8")
        invalid = sorted({ord(character) for character in text if ord(character) < 32 and character not in "\n\r\t"})
        assert invalid == [], f"{path} contains control characters: {invalid}"
