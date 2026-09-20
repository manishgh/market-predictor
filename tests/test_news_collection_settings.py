from pathlib import Path

import pytest

from market_predictor.sources.news_collection_settings import load_news_collection_settings


def _config(path: Path, **changes: object) -> Path:
    values = {
        "owner": "market_predictor", "coordination": "single_host", "storage_root": "receipts",
        "maximum_system_memory_percent": 90.0,
    }
    values.update(changes)
    text = "\n".join(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}" for key, value in values.items())
    path.write_text(text, encoding="utf-8")
    return path


def test_storage_root_is_config_relative_not_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path / "config.toml")
    monkeypatch.chdir(tmp_path.parent)
    settings, root = load_news_collection_settings(config)
    assert settings.owner == "market_predictor"
    assert root == tmp_path / "receipts"
    assert not root.exists()


@pytest.mark.parametrize("changes", [
    {"owner": "trading_flow"}, {"coordination": "distributed"}, {"maximum_system_memory_percent": 91.0},
    {"maximum_system_memory_percent": 0.0}, {"unexpected": "value"}, {"storage_root": ""},
    {"storage_root": "."}, {"storage_root": "//server/share/news"},
])
def test_invalid_deployment_fails_closed(tmp_path: Path, changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        load_news_collection_settings(_config(tmp_path / "config.toml", **changes))


def test_config_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "large.toml"
    path.write_bytes(b" " * 16_385)
    with pytest.raises(ValueError, match="16 KiB"):
        load_news_collection_settings(path)
