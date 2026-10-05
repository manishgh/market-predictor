"""Positive synthetic source-to-publication chain; never retained provider evidence."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_publication import PROFILE
from market_predictor.swing.datasets import issuer_reaction_verification as verifier
from tests.test_issuer_content_qualification_authority import _seal_fixture
from tests.test_issuer_content_qualification_authority import fixture as _source_population
from tests.test_swing_issuer_reaction_publication import reaction_publication as reaction_publication
from tests.test_swing_return_relationship_publication import publication_fixture as publication_fixture

source_population = _source_population


def _month(publication: dict[str, Any], month: str, profile: str) -> pd.DataFrame:
    manifest = json.loads((publication["root"] / publication["publication"].path).read_text(encoding="utf-8"))
    record = manifest["months"][month]["profiles"][profile]
    rows, _ = load_canonical_artifact(publication["output"] / record["path"], allow_research=True)
    return rows


@pytest.fixture
def qualification_fixture(source_population: dict[str, Any], publication_fixture: dict[str, Any]) -> dict[str, Any]:
    """Bind one actual synthetic reviewed version to the native AAA parent issuer.

    The original fixture creates both synthetic independent review files. Resealing
    rebuilds their version/text/sample pins and the blind packets after this edit.
    All other events remain foreign identities and cannot populate AAA or BBB.
    """
    parent = _month(publication_fixture, "2020-06", "technical_relationships")
    security = parent.loc[parent.ticker.eq("AAA"), "security_id"].unique()
    assert len(security) == 1
    identity = str(security[0])
    text = "AAA reports Q2 2020 earnings."
    version_hash = json_sha256({"synthetic_body": text})
    cluster = json_sha256(["earnings-0", identity])
    with sqlite3.connect(source_population["parent"] / "population.sqlite") as connection:
        original = connection.execute(
            "SELECT version_id,content_json FROM versions WHERE json_extract(content_json,'$.source_id')='earnings-0'",
        ).fetchall()
        assert len(original) == 1
        old_id, encoded = original[0]
        row = json.loads(encoded)
        alias = {"synthetic": True, "security_id": identity, "ticker": "AAA",
                 "available_at_utc": "2020-01-01T00:00:00Z"}
        row.update(security_id=identity, ticker="AAA", source_version_sha256=version_hash,
                   text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), alias_proof=alias,
                   identity_authority_sha256=json_sha256(alias))
        assert row["event_available_at_utc"] == row["version_available_at_utc"] == "2020-06-01T20:00:00Z"
        assert row["identity_available_at_utc"] == alias["available_at_utc"]
        connection.execute(
            "UPDATE versions SET version_id=?,cluster_id=?,security_id=?,content_json=?,text=? WHERE version_id=?",
            (json_sha256([cluster, version_hash]), cluster, identity, json.dumps(row, sort_keys=True), text, old_id),
        )
        connection.commit()
    return _seal_fixture(source_population)


def test_native_reviewed_source_populates_completed_reaction_and_preserves_parent(
    reaction_publication: dict[str, Any], qualification_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = reaction_publication
    parent = _month(value["relationship"], "2020-06", "technical_relationships")
    published = _month(value, "2020-06", PROFILE)
    inherited = [name for name in parent.columns if name != "feature_profile"]
    # Includes all 124 inputs/clocks, targets, eligibility, identities and row order.
    pd.testing.assert_frame_equal(published[inherited], parent[inherited], check_exact=True)
    assert published.feature_profile.eq(PROFILE).all()
    selected = published.loc[published.ticker.eq("AAA") & published.session_date_et.eq(date(2020, 6, 2))]
    assert len(selected) == 1
    row = selected.iloc[0]

    root: Path = value["root"]
    stocks = pd.read_parquet(root / "adjusted-archive/unit-AAA/bars.parquet")
    spy = pd.read_parquet(root / "adjusted-archive/unit-SPY/bars.parquet")
    stocks = stocks.sort_values("session_date_et").reset_index(drop=True)
    spy = spy.sort_values("session_date_et").reset_index(drop=True)
    positions = np.flatnonzero(stocks.session_date_et.eq(date(2020, 6, 2)))
    assert len(positions) == 1 and positions[0] >= 20
    position = int(positions[0])
    stock_bar = stocks.iloc[position]
    market = spy.loc[spy.session_date_et.eq(date(2020, 6, 2))]
    assert len(market) == 1
    market_bar = market.iloc[0]
    expected_return = (float(stock_bar.close) / float(stock_bar.open) - 1.0) - (
        float(market_bar.close) / float(market_bar.open) - 1.0
    )
    prior_volumes = stocks.iloc[position - 20:position].volume.to_numpy(dtype=np.float64)
    assert len(prior_volumes) == 20 and np.isfinite(prior_volumes).all()
    expected_volume = float(stock_bar.volume) / float(np.mean(prior_volumes))
    assert expected_return != 0.0
    for name, expected in zip(REACTION_COLUMNS, (expected_return, expected_volume), strict=True):
        assert np.isfinite(row[name])
        assert row[name] == np.float32(expected)
        assert row[f"missing_reason_{name}"] == ""
        assert row[f"available_at_{name}"] == stock_bar.available_at_utc
        assert row[f"available_at_{name}"] <= row.decision_time_utc

    events = pd.read_parquet(qualification_fixture["output"] / "events.parquet")
    event = events.loc[events.event_id.eq("earnings-0")]
    assert len(event) == 1
    qualified = event.iloc[0]
    assert qualified.qualification_status == "qualified" and qualified.event_family == "earnings"
    assert qualified.security_id == row.security_id == stock_bar.security_id
    assert qualified.ticker == row.ticker == "AAA"
    assert row.selected_source_family == "alpaca" and row.selected_event_id == "earnings-0"
    assert row.selected_event_family == "earnings"
    assert row.selected_event_version_sha256 == qualified.event_version_sha256 == json_sha256(
        {"synthetic_body": "AAA reports Q2 2020 earnings."},
    )
    assert row.selected_qualification_authority_sha256 == qualified.qualification_authority_sha256 == file_sha256(
        qualification_fixture["output"] / "_authority.json",
    )
    assert row.selected_event_available_at_utc == qualified.event_available_at_utc == pd.Timestamp("2020-06-01T20:00:00Z")
    assert row.selected_identity_available_at_utc == qualified.identity_available_at_utc == pd.Timestamp("2020-01-01T00:00:00Z")
    assert row.reaction_session_date_et == date(2020, 6, 2)
    assert row.reaction_start_utc == xcals.get_calendar("XNYS").session_open("2020-06-02")
    assert row.reaction_end_utc == stock_bar.bar_end_utc

    # A source observed at June 1's close has no completed reaction that day.
    unfinished = published.loc[published.ticker.eq("AAA") & published.session_date_et.eq(date(2020, 6, 1))]
    foreign = published.loc[published.ticker.eq("BBB") & published.session_date_et.eq(date(2020, 6, 2))]
    assert len(unfinished) == len(foreign) == 1
    for name in REACTION_COLUMNS:
        assert unfinished[name].isna().all() and foreign[name].isna().all()
        assert unfinished[f"missing_reason_{name}"].eq("reaction_session_not_complete_at_decision").all()
        assert foreign[f"missing_reason_{name}"].eq("unknown_source_coverage").all()
        assert unfinished[f"available_at_{name}"].isna().all() and foreign[f"available_at_{name}"].isna().all()

    # Replay all 59 native months with the actual source readers, assembly and
    # qualifier replay. Only fixture-root discovery and resource hooks are scoped;
    # verifier totals use its real manifest/parent/unique-decision equality checks.
    receipt_path = root / "data/reports/issuer_reaction_native_source_chain.json"
    with monkeypatch.context() as scoped:
        scoped.setattr(verifier, "__file__", str(root / "src/market_predictor/swing/datasets/issuer_reaction_verification.py"))
        scoped.setattr(verifier, "_guard", lambda: None)
        scoped.setattr(verifier, "release_process_memory", lambda: None)
        receipt = verifier.verify_issuer_reaction_rows(root, value["publication"], receipt_path)
        verified = verifier.validate_issuer_reaction_receipt(root, value["publication"], receipt)
    assert receipt["status"] == "passed"
    assert receipt["rows"] == receipt["unique_decisions"] == verified.manifest["rows"] == 354
    assert receipt["months"] == len(receipt["monthly"]) == len(verified.months) == 59
    assert receipt["original_columns_exact"] and receipt["original_outcome_values_exact"]
    assert receipt["qualification_source_replayed"] and receipt["additions_source_replayed"]
    assert receipt["event_authority_sha256"] == row.selected_qualification_authority_sha256
    assert receipt["source_files"][value["policy"].qualification_authority.path] == row.selected_qualification_authority_sha256
    totals = receipt["profiles"][PROFILE]
    assert totals["selected_event_rows"] == 2
    assert totals["price_reaction_rows"] == totals["volume_reaction_rows"] == 1
    assert totals["unknown_coverage_rows"] == 354 and totals["clock_violations"] == 0
    assert receipt["report_sha256"] == file_sha256(receipt_path)
    assert all(receipt[name] is False for name in ("training_eligible", "serving_eligible", "promotion_eligible"))
