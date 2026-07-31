from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_materialization import (
    load_complete_swing_feature_panel,
    materialize_swing_feature_panel,
)
from market_predictor.v3.errors import DataReadinessError


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "security_id": ["sec:a", "sec:b", "sec:c", "sec:d"],
            "primary_benchmark": ["XLK"] * 4,
        }
    )


def _rows(group: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for number, row in enumerate(group.itertuples(index=False)):
        for day in (2, 3):
            decision = pd.Timestamp(f"2024-01-{day:02d}T21:01:00Z")
            records.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "session_date_et": pd.Timestamp(f"2024-01-{day:02d}").date(),
                    "decision_time_utc": decision,
                    "available_at_utc": decision,
                    "feature_profile": "technical_market",
                    "feature_eligible": True,
                    "barrier_label": 1,
                    "barrier_label_available_at_utc": decision + pd.Timedelta(days=2),
                    "value": float(number + day),
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def materialization_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, int]]:
    import market_predictor.edge_rebuild.swing_materialization as module

    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "_manifest.json").write_text(
        json.dumps({"artifacts": []}),
        encoding="utf-8",
    )
    memberships_path = tmp_path / "memberships.parquet"
    memberships_path.write_bytes(b"memberships")
    memberships_path.with_suffix(".parquet.manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    memberships = _memberships()
    monkeypatch.setattr(
        module,
        "load_canonical_artifact",
        lambda *_args, **_kwargs: (
            memberships.copy(),
            {"artifact_sha256": "membership-artifact"},
        ),
    )
    monkeypatch.setattr(
        module,
        "load_collection_artifact_index",
        lambda _path: {},
    )
    monkeypatch.setattr(
        module,
        "load_daily_bars",
        lambda ticker, _artifacts: pd.DataFrame({"ticker": [ticker]}),
    )
    monkeypatch.setattr(
        module,
        "load_security_batch_bars",
        lambda group, _artifacts: (group[["ticker"]].copy(), 0),
    )
    calls = {"build": 0, "finalize": 0}

    def build(
        _stocks: pd.DataFrame,
        _benchmarks: pd.DataFrame,
        group: pd.DataFrame,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        calls["build"] += 1
        return _rows(group)

    def finalize(
        rows: pd.DataFrame,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        calls["finalize"] += 1
        return rows.assign(rank_label=1, cross_section_eligible=True)

    monkeypatch.setattr(module, "build_swing_feature_rows", build)
    monkeypatch.setattr(module, "finalize_swing_feature_panel", finalize)
    return collection, memberships_path, calls


def test_materialization_resumes_then_publishes_immutable_panel(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    collection, memberships, calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )

    first = materialize_swing_feature_panel(
        collection_dir=collection,
        memberships_path=memberships,
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
        maximum_stage_one_shards_this_run=1,
    )
    assert first["status"] == "incomplete"
    assert first["completed_stage_one_shards"] == 1
    assert calls == {"build": 1, "finalize": 0}

    complete = materialize_swing_feature_panel(
        collection_dir=collection,
        memberships_path=memberships,
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    assert complete["status"] == "complete"
    assert complete["rows"] == 8
    assert complete["securities"] == 4
    assert complete["stage_one_shards"] == 2
    assert calls == {"build": 2, "finalize": 1}

    replay = materialize_swing_feature_panel(
        collection_dir=collection,
        memberships_path=memberships,
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    assert replay == load_complete_swing_feature_panel(output)
    assert calls == {"build": 2, "finalize": 1}

    with pytest.raises(DataReadinessError, match="resume request differs"):
        materialize_swing_feature_panel(
            collection_dir=collection,
            memberships_path=memberships,
            contract=contract,
            output_dir=output,
            securities_per_shard=1,
        )


def test_resume_refuses_a_corrupted_stage_one_shard(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    collection, memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        collection_dir=collection,
        memberships_path=memberships,
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
        maximum_stage_one_shards_this_run=1,
    )
    with (output / "stage1" / "shard-0000.parquet").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(DataReadinessError, match="does not verify"):
        materialize_swing_feature_panel(
            collection_dir=collection,
            memberships_path=memberships,
            contract=contract,
            output_dir=output,
            securities_per_shard=2,
        )
