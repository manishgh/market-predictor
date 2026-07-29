from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.extended_session_context import (
    build_extended_session_context_plan,
)
from market_predictor.edge_rebuild.history_collection import (
    HISTORY_AUTHORITY_SCHEMA,
    HISTORY_COLLECTION_SCHEMA,
)
from market_predictor.edge_rebuild.history_contracts import (
    EXTENDED_CONTEXT_PLAN_SCHEMA,
    INTRADAY_HISTORY_PLAN_SCHEMA,
    POSTMARKET_SEGMENT,
    PREMARKET_SEGMENT,
    load_extended_session_context_config,
    load_intraday_history_config,
)
from market_predictor.edge_rebuild.intraday_history import (
    EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA,
    PLAN_AUTHORITY_SCHEMA,
    json_sha256,
)
from market_predictor.v3.errors import DataReadinessError

CONTEXT_POLICY = Path("configs/edge_rebuild_extended_session_context.toml")
HISTORY_POLICY = Path("configs/edge_rebuild_intraday_history.toml")
MEMBERSHIPS = Path("data/universe/pit.parquet")
MEMBERSHIP_AUDIT = Path("data/reports/pit_audit.json")
FIRST_SESSION = "2024-07-01"
LAST_SESSION = "2024-07-03"
SESSION_COUNT = 3
FROZEN_ER1A_POLICY_SHA256 = (
    "252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830"
)


def test_frozen_er1a_policy_identity_is_unchanged() -> None:
    """The published ER1A plan records this hash; refactors may not move it."""

    config = load_intraday_history_config(HISTORY_POLICY)

    assert config.sha256() == FROZEN_ER1A_POLICY_SHA256


def test_context_plan_covers_both_segments_without_regular_session_rows(
    tmp_path: Path,
) -> None:
    plan_dir, collection_dir = _write_regular_layer(tmp_path)
    _write_universe(tmp_path)

    manifest = build_extended_session_context_plan(
        intraday_plan_directory=plan_dir,
        intraday_collection_directory=collection_dir,
        memberships_path=tmp_path / MEMBERSHIPS,
        membership_audit_path=tmp_path / MEMBERSHIP_AUDIT,
        policy_path=CONTEXT_POLICY,
        output_directory=tmp_path / "context_plan",
        config=load_extended_session_context_config(CONTEXT_POLICY),
    )

    summary = manifest["summary"]
    assert manifest["schema"] == EXTENDED_CONTEXT_PLAN_SCHEMA
    assert manifest["acquisition"]["regular_session_rows_refetched"] == 0
    assert summary["planned_history_sessions"] == SESSION_COUNT
    assert summary["premarket_units"] == summary["postmarket_units"]
    assert summary["acquisition_units"] == 2 * summary["premarket_units"]

    units = _read_units(tmp_path / "context_plan")
    windows = _read_windows(tmp_path / "context_plan")
    assert set(units["session_segment"]) == {
        PREMARKET_SEGMENT,
        POSTMARKET_SEGMENT,
    }
    assert not bool(units["unit_id"].duplicated().any())
    assert set(units["timeframe"]) == {"5Min"}
    assert set(units["price_feed"]) == {"sip"}
    assert set(units["adjustment"]) == {"all"}

    premarket = units[units["session_segment"] == PREMARKET_SEGMENT]
    postmarket = units[units["session_segment"] == POSTMARKET_SEGMENT]
    assert bool(
        premarket["requested_end_utc"]
        .isin(set(windows["session_open_utc"]))
        .all()
    )
    assert bool(
        postmarket["requested_start_utc"]
        .isin(set(windows["session_close_utc"]))
        .all()
    )
    # No planned request may reach into the frozen regular session, so the
    # extended layer can never re-fetch or contradict an ER1A bar.
    assert bool(
        premarket["requested_end_utc"].le(premarket["session_open_utc"]).all()
    )
    assert bool(
        postmarket["requested_start_utc"]
        .ge(postmarket["session_close_utc"])
        .all()
    )
    assert bool(
        premarket["requested_start_utc"]
        .lt(premarket["session_open_utc"])
        .all()
    )
    assert bool(
        postmarket["requested_end_utc"].gt(postmarket["session_close_utc"]).all()
    )


def test_context_plan_binds_to_the_frozen_regular_session_layer(
    tmp_path: Path,
) -> None:
    plan_dir, collection_dir = _write_regular_layer(tmp_path)
    _write_universe(tmp_path)
    foreign_plan, _ = _write_regular_layer(
        tmp_path,
        suffix="_other",
        first_session="2024-06-03",
    )

    with pytest.raises(DataReadinessError, match="does not belong"):
        build_extended_session_context_plan(
            intraday_plan_directory=foreign_plan,
            intraday_collection_directory=collection_dir,
            memberships_path=tmp_path / MEMBERSHIPS,
            membership_audit_path=tmp_path / MEMBERSHIP_AUDIT,
            policy_path=CONTEXT_POLICY,
            output_directory=tmp_path / "context_plan_foreign",
            config=load_extended_session_context_config(CONTEXT_POLICY),
        )
    assert not (tmp_path / "context_plan_foreign").exists()
    assert plan_dir.exists()


def test_context_plan_rejects_a_reused_output_directory(
    tmp_path: Path,
) -> None:
    plan_dir, collection_dir = _write_regular_layer(tmp_path)
    _write_universe(tmp_path)
    output = tmp_path / "context_plan"
    output.mkdir()

    with pytest.raises(DataReadinessError, match="must be new"):
        build_extended_session_context_plan(
            intraday_plan_directory=plan_dir,
            intraday_collection_directory=collection_dir,
            memberships_path=tmp_path / MEMBERSHIPS,
            membership_audit_path=tmp_path / MEMBERSHIP_AUDIT,
            policy_path=CONTEXT_POLICY,
            output_directory=output,
            config=load_extended_session_context_config(CONTEXT_POLICY),
        )


def test_context_plan_publishes_verifiable_authority(tmp_path: Path) -> None:
    plan_dir, collection_dir = _write_regular_layer(tmp_path)
    _write_universe(tmp_path)
    output = tmp_path / "context_plan"

    manifest = build_extended_session_context_plan(
        intraday_plan_directory=plan_dir,
        intraday_collection_directory=collection_dir,
        memberships_path=tmp_path / MEMBERSHIPS,
        membership_audit_path=tmp_path / MEMBERSHIP_AUDIT,
        policy_path=CONTEXT_POLICY,
        output_directory=output,
        config=load_extended_session_context_config(CONTEXT_POLICY),
    )

    authority = json.loads((output / "_authority.json").read_text("utf-8"))
    assert authority["schema"] == EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA
    assert authority["artifact_sha256"] == file_sha256(output / "_manifest.json")
    assert authority["plan_fingerprint"] == manifest["plan_fingerprint"]
    for record in manifest["files"]:
        path = output / str(record["path"])
        assert file_sha256(path) == record["sha256"]


def _write_universe(root: Path) -> None:
    tickers = [f"T{index:03d}" for index in range(460)]
    # One closed interval proves the universe carries real exits, not only
    # open-ended rows; the verifier rejects a universe without any.
    effective_to = [pd.Timestamp("2030-01-01", tz="UTC")] + [pd.NaT] * (
        len(tickers) - 1
    )
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "security_id": [f"cik:{index:07d}" for index in range(len(tickers))],
            "effective_from_utc": pd.Timestamp("2019-07-09", tz="UTC"),
            "effective_to_utc": pd.to_datetime(pd.Series(effective_to), utc=True),
            "sector": "Industrials",
            "primary_benchmark": "XLI",
            "universe_snapshot_id": "test-pit-snapshot",
            "membership_source_urls": '["https://example.invalid/sp500"]',
        }
    )
    path = root / MEMBERSHIPS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    audit_path = root / MEMBERSHIP_AUDIT
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "universe_snapshot_id": "test-pit-snapshot",
                "historical_tickers": len(tickers),
                "membership_intervals": len(frame),
                "contradictions": [],
            }
        ),
        encoding="utf-8",
    )


def _write_regular_layer(
    root: Path,
    *,
    suffix: str = "",
    first_session: str = FIRST_SESSION,
) -> tuple[Path, Path]:
    if not (root / MEMBERSHIPS).exists():
        _write_universe(root)
    plan_dir = root / f"regular_plan{suffix}"
    collection_dir = root / f"regular_collection{suffix}"
    request: dict[str, Any] = {
        "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
        "variant": suffix,
        "membership": {
            "path": str(root / MEMBERSHIPS),
            "audit_path": str(root / MEMBERSHIP_AUDIT),
            "sha256": file_sha256(root / MEMBERSHIPS),
            "universe_snapshot_id": "test-pit-snapshot",
        },
    }
    fingerprint = json_sha256(request)
    plan_dir.mkdir(parents=True)
    request_path = plan_dir / "_request.json"
    request_path.write_text(
        json.dumps({**request, "plan_fingerprint": fingerprint}, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
        "plan_fingerprint": fingerprint,
        "policy_sha256": load_intraday_history_config(HISTORY_POLICY).sha256(),
        "summary": {
            "first_history_session": first_session,
            "last_history_session": LAST_SESSION,
            "planned_history_sessions": SESSION_COUNT,
        },
        "files": [
            {
                "path": "_request.json",
                "sha256": file_sha256(request_path),
                "bytes": request_path.stat().st_size,
                "rows": 1,
            }
        ],
    }
    (plan_dir / "_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (plan_dir / "_authority.json").write_text(
        json.dumps(
            {
                "schema": PLAN_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(plan_dir / "_manifest.json"),
                "plan_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    _write_collection(collection_dir, fingerprint)
    return plan_dir, collection_dir


def _write_collection(directory: Path, plan_fingerprint: str) -> None:
    directory.mkdir(parents=True)
    request = {
        "schema": HISTORY_COLLECTION_SCHEMA,
        "plan_fingerprint": plan_fingerprint,
    }
    request_sha256 = json_sha256(request)
    (directory / "_request.json").write_text(
        json.dumps({**request, "request_sha256": request_sha256}, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "_manifest.json").write_text(
        json.dumps(
            {
                "schema": HISTORY_COLLECTION_SCHEMA,
                "status": "transport_complete",
                "request_sha256": request_sha256,
                "plan_fingerprint": plan_fingerprint,
                "total_rows": 1_234,
                "artifacts": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (directory / "_authority.json").write_text(
        json.dumps(
            {
                "schema": HISTORY_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(directory / "_manifest.json"),
                "request_sha256": request_sha256,
                "plan_fingerprint": plan_fingerprint,
            }
        ),
        encoding="utf-8",
    )


def _read_units(directory: Path) -> pd.DataFrame:
    parts = sorted((directory / "units" / "5Min").glob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)


def _read_windows(directory: Path) -> pd.DataFrame:
    parts = sorted((directory / "session_windows").glob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
