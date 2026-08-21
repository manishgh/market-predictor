from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import market_predictor.edge_rebuild.sp500_transitions as transition_module
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.v3.spglobal_archive import VerifiedSpGlobalRawArchive


def test_official_fox_transitions_are_one_atomic_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path)
    reviewed = _reviewed_ledger(tmp_path)
    monkeypatch.setattr(
        transition_module,
        "_verified_parents",
        lambda **_: (archive, _parent_lineage()),
    )
    monkeypatch.setattr(
        transition_module,
        "read_verified_spglobal_release_html",
        lambda *_: _fox_html(),
    )

    manifest = transition_module.publish_sp500_transition_authority(
        archive_directory=archive.root,
        event_directory=tmp_path / "events",
        reviewed_transitions_path=reviewed,
        start_date=date(2018, 5, 29),
        cutoff_date=date(2026, 7, 8),
        output_directory=tmp_path / "transitions",
    )

    assert manifest["transition_count"] == 4
    transitions = pd.read_parquet(tmp_path / "transitions" / "transitions.parquet")
    assert set(zip(transitions["old_ticker"], transitions["new_ticker"], strict=True)) == {
        ("FOX", "TFCF"),
        ("FOXA", "TFCFA"),
        ("FOXAV", "FOXA"),
        ("FOXBV", "FOX"),
    }
    assert transitions["atomic_group_id"].nunique() == 1


def test_ambiguous_same_time_chain_fails_closed() -> None:
    moment = pd.Timestamp("2019-03-19T04:00:00Z")
    rows = pd.DataFrame(
        {
            "transition_id": ["one", "two"],
            "effective_at_utc": [moment, moment],
            "old_ticker": ["AAA", "BBB"],
            "new_ticker": ["BBB", "CCC"],
            "identity_continuity": [True, True],
            "source_kind": ["reviewed_transition_ledger"] * 2,
            "source_sha256": ["a" * 64] * 2,
            "atomic_group_id": ["one", "two"],
        }
    )

    with pytest.raises(DataReadinessError, match="ambiguous same-time"):
        transition_module._validate_transition_graph(rows)


def test_transition_poison_cannot_be_hidden_by_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path)
    reviewed = _reviewed_ledger(tmp_path)
    monkeypatch.setattr(
        transition_module,
        "_verified_parents",
        lambda **_: (archive, _parent_lineage()),
    )
    monkeypatch.setattr(
        transition_module,
        "read_verified_spglobal_release_html",
        lambda *_: _fox_html(),
    )
    output = tmp_path / "transitions"
    transition_module.publish_sp500_transition_authority(
        archive_directory=archive.root,
        event_directory=tmp_path / "events",
        reviewed_transitions_path=reviewed,
        start_date=date(2018, 5, 29),
        cutoff_date=date(2026, 7, 8),
        output_directory=output,
    )
    path = output / "transitions.parquet"
    poisoned = pd.read_parquet(path)
    poisoned.loc[0, "new_ticker"] = "FAKE"
    poisoned.to_parquet(path, index=False)
    manifest = _object(output / "_manifest.json")
    manifest["transition_artifact"] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    transition_module._write_json_atomic(output / "_manifest.json", manifest)
    authority = _object(output / "_authority.json")
    authority["artifact_sha256"] = file_sha256(output / "_manifest.json")
    transition_module._write_json_atomic(output / "_authority.json", authority)

    with pytest.raises(DataReadinessError, match="does not replay"):
        transition_module.require_sp500_transition_authority(
            output,
            archive_directory=archive.root,
            event_directory=tmp_path / "events",
            reviewed_transitions_path=reviewed,
            start_date=date(2018, 5, 29),
            cutoff_date=date(2026, 7, 8),
        )


def _archive(tmp_path: Path) -> VerifiedSpGlobalRawArchive:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "_authority.json").write_text("{}", encoding="utf-8")
    return VerifiedSpGlobalRawArchive(
        root=root,
        authority={"artifact_sha256": "a" * 64},
        manifest={"release_set_sha256": "b" * 64},
        releases=(
            {
                "url": "https://press.spglobal.com/2019-03-14-fox",
                "published_date": "2019-03-14",
                "sha256": "c" * 64,
                "unit_id": "fox",
            },
        ),
    )


def _reviewed_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "id,effective_date,old_symbol,new_symbol,transition_type,"
        "identity_continuity,membership_continuity,old_security_id,"
        "new_security_id,source_url,evidence_summary,reviewed_by,reviewed_at_utc\n"
        "later,2027-01-01,AAA,BBB,name_change,true,true,sec:a,sec:a,"
        "https://example.com/evidence,reviewed evidence,tester,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    return path


def _parent_lineage() -> dict[str, str]:
    return {
        "raw_authority_sha256": "1" * 64,
        "raw_manifest_sha256": "2" * 64,
        "raw_release_set_sha256": "3" * 64,
        "event_authority_sha256": "4" * 64,
        "event_set_sha256": "5" * 64,
    }


def _fox_html() -> str:
    return """
    <html><body><div class="wd_news_body">
    <p>Effective on March 19, Twenty-First Century Fox will change its
    Class A common stock ticker from FOXA to TFCFA and its Class B common
    stock ticker from FOX to TFCF. Also effective on March 19, Fox Corp.
    will change its Class A common stock ticker from FOXAV to FOXA and its
    Class B common stock ticker from FOXBV to FOX.</p>
    </div></body></html>
    """


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
