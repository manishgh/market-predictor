from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_predictor.edge_rebuild.sec_identity_authority as identity
from market_predictor.core.errors import DataReadinessError


def _config() -> identity.SecIdentityConfig:
    return identity.SecIdentityConfig()


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _member("security:rename", "OLD", "2019-07-09", "2022-07-05"),
            _member("security:rename", "NEW", "2022-07-05", None),
            _member("security:class-a", "GOOG", "2019-07-09", None),
            _member("security:class-c", "GOOGL", "2019-07-09", None),
        ]
    )


def _member(security_id: str, ticker: str, start: str, end: str | None) -> dict[str, object]:
    start_utc = pd.Timestamp(start, tz="America/New_York").tz_convert("UTC").isoformat()
    end_utc = None if end is None else pd.Timestamp(end, tz="America/New_York").tz_convert("UTC").isoformat()
    return {
        "security_id": security_id,
        "ticker": ticker,
        "effective_from_utc": start_utc,
        "effective_to_utc": end_utc,
        "available_at_utc": start_utc,
    }


def _transitions(*, proven: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "effective_at_utc": "2022-07-05T04:00:00Z",
                "old_ticker": "OLD",
                "new_ticker": "NEW",
                "identity_continuity": proven,
                "old_security_id": "security:rename",
                "new_security_id": "security:rename",
            }
        ]
    )


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "NEW", "sec_cik": "1", "company_name": "Renamed Inc"},
            {"ticker": "GOOG", "sec_cik": "1652044", "company_name": "Alphabet Inc"},
            {"ticker": "GOOGL", "sec_cik": "1652044", "company_name": "Alphabet Inc"},
        ]
    )


def _write_mapping(path: Path, *, include_extra: bool = False) -> None:
    payload: dict[str, object] = {
        "0": {"cik_str": 1, "ticker": "NEW", "title": "Renamed Inc"},
        "1": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc"},
        "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc"},
    }
    if include_extra:
        payload["3"] = {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _override(
    *,
    security_id: str = "cik:0000000042",
    ticker: str = "GONE",
    cik: str = "42",
    start: str = "2019-07-09T04:00:00Z",
    end: str = "2026-07-09T04:00:00Z",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": security_id,
                "ticker": ticker,
                "issuer_name": "Gone Inc",
                "sec_cik": cik,
                "effective_from_utc": start,
                "effective_to_utc": end,
                "evidence_url": "https://www.sec.gov/Archives/edgar/data/42/000000004224000001/proof.htm",
                "evidence_document": "proof.htm",
                "evidence_accession": "0000000042-24-000001",
                "evidence_raw_sha256": "a" * 64,
                "reviewer_status": "approved",
                "reason": "official filing cover lists trading symbol",
            }
        ]
    )


def _write_overrides(path: Path, frame: pd.DataFrame | None = None) -> None:
    columns = list(identity._OVERRIDE_COLUMNS)
    (frame if frame is not None else pd.DataFrame(columns=columns)).to_csv(path, index=False, lineterminator="\n")


def test_relation_propagates_only_within_stable_security_and_supports_dual_class() -> None:
    relations, coverage, summary = identity.build_sec_identity_relations(_memberships(), _transitions(), _mapping(), config=_config())

    rename = relations.loc[relations["security_id"].eq("security:rename")]
    assert rename["ticker"].tolist() == ["OLD", "NEW"]
    assert rename["sec_cik"].unique().tolist() == ["0000000001"]
    alphabet = relations.loc[relations["security_id"].str.startswith("security:class")]
    assert alphabet["sec_cik"].unique().tolist() == ["0001652044"]
    assert alphabet["security_id"].nunique() == 2
    assert summary["dual_class_cik_count"] == 1
    assert summary["coverage_passed"] is True
    assert coverage["status"].eq("resolved").all()


def test_ticker_reuse_and_missing_or_unproven_identity_remain_excluded() -> None:
    memberships = pd.concat(
        [
            _memberships(),
            pd.DataFrame(
                [
                    _member("security:reuse-old", "USED", "2019-07-09", "2021-01-04"),
                    _member("security:reuse-new", "USED", "2021-01-04", None),
                    _member("security:missing", "GONE", "2019-07-09", None),
                    _member("cik:0000000010", "CONFLICT", "2019-07-09", None),
                ]
            ),
        ],
        ignore_index=True,
    )
    mapping = pd.concat(
        [
            _mapping(),
            pd.DataFrame(
                [
                    {"ticker": "USED", "sec_cik": "9", "company_name": "Current User"},
                    {"ticker": "CONFLICT", "sec_cik": "11", "company_name": "Wrong Issuer"},
                ]
            ),
        ],
        ignore_index=True,
    )
    _, coverage, summary = identity.build_sec_identity_relations(memberships, _transitions(proven=False), mapping, config=_config())
    reasons = coverage.set_index("security_id")["reason"].to_dict()

    assert reasons["security:rename"] == "ticker_transition_identity_not_proven"
    assert reasons["security:reuse-old"] == "ticker_reused_by_different_security_id"
    assert reasons["security:reuse-new"] == "ticker_reused_by_different_security_id"
    assert reasons["security:missing"] == "latest_ticker_absent_from_sec_mapping"
    assert reasons["cik:0000000010"] == "stable_security_id_cik_conflicts_with_sec_mapping"
    assert summary["coverage_passed"] is False
    assert summary["unresolved_security_ids"] == sorted(
        [
            "security:missing",
            "security:rename",
            "security:reuse-new",
            "security:reuse-old",
            "cik:0000000010",
        ]
    )


def test_raw_mapping_normalizes_sec_share_class_symbols(tmp_path: Path) -> None:
    path = tmp_path / "company_tickers.json"
    path.write_text(
        json.dumps(
            {
                "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
                "1": {"cik_str": 1067983, "ticker": "BRK-A", "title": "Berkshire"},
            }
        ),
        encoding="utf-8",
    )

    mapping = identity.load_sec_company_ticker_mapping(path)

    assert mapping["ticker"].tolist() == ["BRK.A", "BRK.B"]
    assert mapping["sec_cik"].unique().tolist() == ["0001067983"]


def test_reviewed_override_resolves_absent_ticker_with_exact_security_interval() -> None:
    memberships = pd.DataFrame([_member("cik:0000000042", "GONE", "2019-07-09", None)])

    relations, coverage, summary = identity.build_sec_identity_relations(
        memberships,
        _transitions().iloc[0:0],
        _mapping(),
        _override(),
        config=_config(),
    )

    assert relations.loc[0, "sec_cik"] == "0000000042"
    assert relations.loc[0, "identity_policy"] == "reviewed_official_sec_filing_override_v1"
    assert coverage.loc[0, "status"] == "resolved"
    assert summary["coverage_passed"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"ticker": "OTHER"}, "ticker mismatch"),
        ({"sec_cik": "43"}, "embedded CIK"),
        ({"effective_to_utc": "2025-01-01T05:00:00Z"}, "interval mismatch"),
    ],
)
def test_reviewed_override_fails_closed_on_identity_mismatch(change: dict[str, str], message: str) -> None:
    memberships = pd.DataFrame([_member("cik:0000000042", "GONE", "2019-07-09", None)])
    override = _override().assign(**change)

    with pytest.raises(DataReadinessError, match=message):
        identity.build_sec_identity_relations(
            memberships,
            _transitions().iloc[0:0],
            _mapping(),
            override,
            config=_config(),
        )


def test_reviewed_override_loader_verifies_official_url_and_raw_hash(tmp_path: Path) -> None:
    evidence = tmp_path / "proof.htm"
    evidence.write_text("Trading Symbol GONE", encoding="utf-8")
    ledger = tmp_path / "overrides.csv"
    frame = _override()
    frame.loc[0, "evidence_raw_sha256"] = identity.file_sha256(evidence)
    _write_overrides(ledger, frame)

    loaded = identity.load_reviewed_sec_identity_overrides(ledger)

    assert loaded.loc[0, "security_id"] == "cik:0000000042"
    evidence.write_text("tampered", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        identity.load_reviewed_sec_identity_overrides(ledger)


def test_reviewed_override_rejects_ticker_reuse() -> None:
    memberships = pd.DataFrame(
        [
            _member("cik:0000000042", "GONE", "2019-07-09", "2022-01-03"),
            _member("security:new", "GONE", "2022-01-03", None),
        ]
    )
    with pytest.raises(DataReadinessError, match="ticker is reused"):
        identity.build_sec_identity_relations(
            memberships,
            _transitions().iloc[0:0],
            _mapping(),
            _override(end="2022-01-03T05:00:00Z"),
            config=_config(),
        )


def test_config_freezes_window_and_five_percent_policy(tmp_path: Path) -> None:
    assert identity.load_sec_identity_config(Path("configs/edge_rebuild_sec_identity.toml")) == _config()
    with pytest.raises(DataReadinessError, match="frozen at 5%"):
        identity.validate_sec_identity_config(identity.SecIdentityConfig(maximum_whole_security_exclusion_fraction=0.051))


def test_publish_replays_and_detects_mapping_or_artifact_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = tmp_path / "company_tickers.json"
    _write_mapping(mapping)
    membership_dir = tmp_path / "memberships"
    transition_dir = tmp_path / "transitions"
    membership_dir.mkdir()
    transition_dir.mkdir()
    (membership_dir / "_authority.json").write_text(json.dumps({"universe_sha256": "a" * 64}), encoding="utf-8")
    (transition_dir / "_authority.json").write_text(json.dumps({"transition_set_sha256": "b" * 64}), encoding="utf-8")
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("review evidence\n", encoding="utf-8")
    overrides = tmp_path / "overrides.csv"
    _write_overrides(overrides)
    monkeypatch.setattr(identity, "require_sp500_membership_authority", lambda *args, **kwargs: _memberships())
    monkeypatch.setattr(identity, "require_sp500_transition_authority", lambda *args, **kwargs: _transitions())
    common = {
        "sec_mapping_path": mapping,
        "reviewed_overrides_path": overrides,
        "membership_directory": membership_dir,
        "archive_directory": tmp_path / "archive",
        "event_directory": tmp_path / "events",
        "transition_directory": transition_dir,
        "reviewed_transitions_path": reviewed,
        "anchor_path": tmp_path / "anchor.csv",
        "config": _config(),
    }
    output = tmp_path / "identity"

    published = identity.publish_sec_identity_authority(**common, output_directory=output)

    assert len(published.relations) == 4
    assert published.manifest["coverage"]["excluded_security_count"] == 0
    identity.require_sec_identity_authority(output, **common)
    _write_mapping(mapping, include_extra=True)
    with pytest.raises(DataReadinessError, match="request or parent lineage"):
        identity.require_sec_identity_authority(output, **common)
    _write_mapping(mapping)
    overrides.write_text(overrides.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="request or parent lineage"):
        identity.require_sec_identity_authority(output, **common)
    _write_overrides(overrides)
    relation_path = output / identity.RELATION_FILE
    relation_path.write_bytes(relation_path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="artifact hash"):
        identity.require_sec_identity_authority(output, **common)


def test_publication_refuses_more_than_five_percent_whole_security_exclusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memberships = _memberships()
    memberships = pd.concat(
        [memberships, pd.DataFrame([_member("security:missing", "GONE", "2019-07-09", None)])],
        ignore_index=True,
    )
    mapping = tmp_path / "company_tickers.json"
    _write_mapping(mapping)
    membership_dir = tmp_path / "memberships"
    transition_dir = tmp_path / "transitions"
    membership_dir.mkdir()
    transition_dir.mkdir()
    (membership_dir / "_authority.json").write_text("{}", encoding="utf-8")
    (transition_dir / "_authority.json").write_text("{}", encoding="utf-8")
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("review evidence\n", encoding="utf-8")
    overrides = tmp_path / "overrides.csv"
    _write_overrides(overrides)
    monkeypatch.setattr(identity, "require_sp500_membership_authority", lambda *args, **kwargs: memberships)
    monkeypatch.setattr(identity, "require_sp500_transition_authority", lambda *args, **kwargs: _transitions())

    with pytest.raises(DataReadinessError, match="above the frozen 5.00% ceiling"):
        identity.publish_sec_identity_authority(
            sec_mapping_path=mapping,
            reviewed_overrides_path=overrides,
            membership_directory=membership_dir,
            archive_directory=tmp_path / "archive",
            event_directory=tmp_path / "events",
            transition_directory=transition_dir,
            reviewed_transitions_path=reviewed,
            anchor_path=tmp_path / "anchor.csv",
            output_directory=tmp_path / "identity",
            config=_config(),
        )
    assert not (tmp_path / "identity").exists()
