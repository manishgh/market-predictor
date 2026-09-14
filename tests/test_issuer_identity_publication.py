"""Synthetic issuer text and authorities only; no fixture is production evidence."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.catalysts.issuer_events.attribution import build_event_security_relations
from market_predictor.catalysts.issuer_events.identity_publication import (
    PUBLICATION_POLICY,
    _document_bodies,
    _publication_proxy,
    build_issuer_identity_inputs,
    publish_issuer_identity_authority,
    validate_issuer_event_scope,
    verify_issuer_identity_inputs,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.sentiment_history import _metadata_by_security_and_ticker, _read_universe

_CONFIG = Path(__file__).resolve().parents[1] / "configs/swing_issuer_identity_authority.toml"


@pytest.fixture
def issuer_inputs(tmp_path: Path):  # type: ignore[no-untyped-def]
    scope = tmp_path / "configs/swing_issuer_news_corrections.json"
    scope.parent.mkdir()
    scope.write_text("synthetic test-only scope", encoding="utf-8")
    (tmp_path / "inventory").write_text("synthetic test-only inventory", encoding="utf-8")
    config = tmp_path / "issuer.toml"
    policy_text = _CONFIG.read_text(encoding="utf-8")
    policy = tomllib.loads(policy_text)
    correction_path = tmp_path / policy["correction_policy"]
    correction_text = (_CONFIG.parent / "swing_symbol_corrections.toml").read_text(encoding="utf-8")
    correction = tomllib.loads(correction_text)
    correction_path.write_text(correction_text.replace(
        f'document_inventory = "{correction["document_inventory"]}"', 'document_inventory = "inventory"'
    ).replace(
        f'document_archive = "{correction["document_archive"]}"', 'document_archive = "original"'
    ), encoding="utf-8")
    config.write_text(policy_text.replace(policy["query_scope_sha256"], file_sha256(scope)).replace(
        policy["correction_policy_sha256"], file_sha256(correction_path)), encoding="utf-8")
    bodies = {
        "echostar_pre_window_symbol": (
            b"<p>SYNTHETIC TEST ONLY. As filed with the Securities and Exchange Commission on June 6, 2019. "
            b"EchoStar Corporation (EchoStar) Class A common stock SATS.</p>"),
        "fiserv_pre_window_symbol": b"<p>SYNTHETIC TEST ONLY. Fiserv, Inc. Common Stock, par value $0.01 per share FISV.</p>",
        "fiserv_pre_window_filing_index": (
            b"<p>SYNTHETIC TEST ONLY. Filing Date 2019-07-01 Accepted 2019-07-01 08:02:36 "
            b"CIK : 0000798354 d15278d8k.htm</p>"),
    }
    intervals = pd.DataFrame([
        {"security_id": "cik:0001415404", "ticker": "SATS", "effective_from_utc": "2019-07-09T00:00:00Z",
            "effective_to_utc": "2024-05-29T00:00:00Z"},
        {"security_id": "cik:0000798354", "ticker": "FISV", "effective_from_utc": "2019-07-09T00:00:00Z",
            "effective_to_utc": "2023-06-07T13:30:00Z"},
    ])
    namespace = "market_predictor.catalysts.issuer_events.identity_publication."

    def documents(root, inventory, archive, pin, wanted):  # type: ignore[no-untyped-def]
        return {key: value for key, value in bodies.items() if key in wanted and (key.endswith("index") != (archive == "original"))}

    with patch(namespace + "load_news_query_scope", return_value=(intervals, {})), \
        patch(namespace + "_document_bodies", side_effect=documents), \
        patch("market_predictor.sources.official_documents.load_official_document_inventory", return_value=SimpleNamespace(documents=[
            SimpleNamespace(document_id="fiserv_pre_window_symbol", url="https://www.sec.gov/test/d15278d8k.htm")])):
        yield tmp_path, config, bodies


def test_publication_proxy_uses_filing_date_not_report_date_or_capture() -> None:
    for body, kind in ((b"<p>Date of Report July 1, 2019</p>", "as_filed_cover"),
        (b"<p>Period of Report 2019-07-01 Accepted 2019-07-01 08:02:36</p>", "sec_filing_index")):
        with pytest.raises(DataReadinessError, match="date"):
            _publication_proxy(body, kind)
    day, available = _publication_proxy(b"<p>Filing Date 2019-07-01 Accepted 2019-07-01 08:02:36</p>", "sec_filing_index")
    assert day == "2019-07-01"
    assert available == pd.Timestamp("2019-07-02T04:00:00Z")
    _, winter = _publication_proxy(b"<p>Filing Date 2019-01-02 Accepted 2019-01-02 08:02:36</p>", "sec_filing_index")
    assert winter == pd.Timestamp("2019-01-03T05:00:00Z")


def test_identity_only_publication_roundtrips_without_business_facts(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, _ = issuer_inputs
    output = root / "authority"
    result = publish_issuer_identity_authority(root=root, config=config,
        expected_config_sha256=file_sha256(config), output_directory=output)
    labels, lm = load_canonical_artifact(Path(result["business_labels"]), allow_research=True)
    identities, im = load_canonical_artifact(Path(result["security_identities"]), allow_research=True)
    verify_issuer_identity_inputs(labels, identities, lm, im)
    assert im["artifact_path"] == result["security_identities"]
    assert labels.empty
    assert identities["security_id"].tolist() == ["cik:0000798354", "cik:0001415404"]
    assert identities["availability_policy"].eq(PUBLICATION_POLICY).all()
    assert identities["historical_disposition"].eq("unknown_business_segments").all()
    assert not identities["historical_first_seen_proven"].any()
    assert not identities["source_coverage_admitted"].any()
    assert identities["available_at_utc"].tolist() == [pd.Timestamp("2019-07-02T04:00:00Z"), pd.Timestamp("2019-06-07T04:00:00Z")]
    assert not lm["production_ready"] and not im["production_ready"]
    with pytest.raises(DataReadinessError, match="immutable"):
        publish_issuer_identity_authority(root=root, config=config,
            expected_config_sha256=file_sha256(config), output_directory=output)
    identities.loc[0, "company"] = "Synthetic Unsupported Owner"
    with pytest.raises(DataReadinessError, match="reproduce"):
        verify_issuer_identity_inputs(labels, identities, lm, im)


def test_sentiment_reads_verified_identity_only_authority(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, bodies = issuer_inputs
    result = publish_issuer_identity_authority(root=root, config=config,
        expected_config_sha256=file_sha256(config), output_directory=root / "sentiment-identities")
    path = Path(result["security_identities"])
    frame = _read_universe(path)
    assert frame["sector"].eq("").all() and frame["industry"].eq("").all()
    assert not frame["source_coverage_admitted"].any()
    metadata = _metadata_by_security_and_ticker(frame, identity_only=True)
    assert metadata[("cik:0001415404", "SATS")].company == "EchoStar Corporation"
    good = pd.DataFrame([{"security_id": "cik:0000798354", "ticker": "FISV",
        "feature_available_at_utc": "2023-06-07T13:29:59Z"}])
    validate_issuer_event_scope(good, frame, frame.attrs["issuer_identity_manifest"])
    good["feature_available_at_utc"] = "2023-06-07T13:30:00Z"
    with pytest.raises(DataReadinessError, match="outside"):
        validate_issuer_event_scope(good, frame, frame.attrs["issuer_identity_manifest"])
    bodies["fiserv_pre_window_symbol"] = b"SYNTHETIC TEST ONLY: unsupported owner"
    with pytest.raises(DataReadinessError, match="absent"):
        _read_universe(path)


def test_official_company_only_matching_and_exact_half_open_cutoffs(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, _ = issuer_inputs
    labels, identities, _ = build_issuer_identity_inputs(root=root, config=config, expected_config_sha256=file_sha256(config))
    for sid, ticker, company, end in (("cik:0000798354", "FISV", "Fiserv", "2023-06-07T13:30:00Z"),
        ("cik:0001415404", "SATS", "EchoStar", "2024-05-29T00:00:00Z")):
        boundary = pd.Timestamp(end)
        events = pd.DataFrame([
            {"event_id": str(index), "security_id": sid, "ticker": ticker,
                "feature_available_at_utc": timestamp, "title": title, "summary": "", "text": ""}
            for index, (timestamp, title) in enumerate([
                (pd.Timestamp("2019-07-08T23:59:59Z"), f"{company} reports earnings"),
                (pd.Timestamp("2019-07-09T00:00:00Z"), f"{company} reports earnings"),
                (boundary - pd.Timedelta(nanoseconds=1), f"{company} reports earnings"),
                (boundary, f"{company} reports earnings"),
                (boundary - pd.Timedelta(seconds=1), "Unrelated sector news"),
            ])
        ])
        relations = build_event_security_relations(events, labels, identities)
        assert relations["event_id"].tolist() == ["1", "2"]
        assert relations["target_security_id"].eq(sid).all()
        assert relations["relation_channel"].eq("direct_issuer").all()


@pytest.mark.parametrize("poison", ["company", "class", "publication", "cik", "filing"])
def test_missing_or_competing_issuer_evidence_fails_closed(issuer_inputs, poison: str) -> None:  # type: ignore[no-untyped-def]
    root, config, bodies = issuer_inputs
    key = "fiserv_pre_window_symbol" if poison in {"company", "class"} else "fiserv_pre_window_filing_index"
    old, new = {"company": (b"Fiserv, Inc.", b"Different Owner"), "class": (b"Common Stock", b"Debt Notes"),
        "publication": (b"2019-07-01", b"2019-07-09"), "cik": (b"0000798354", b"0000000001"),
        "filing": (b"d15278d8k.htm", b"another.htm")}[poison]
    bodies[key] = bodies[key].replace(old, new)
    with pytest.raises(DataReadinessError):
        build_issuer_identity_inputs(root=root, config=config, expected_config_sha256=file_sha256(config))


def test_policy_tampering_is_rejected_before_source_loading(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, _ = issuer_inputs
    with pytest.raises(DataReadinessError, match="independent reviewed pin"):
        build_issuer_identity_inputs(root=root, config=config, expected_config_sha256="f" * 64)


def test_ticker_text_cannot_bypass_reviewed_identity_or_dates(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, _ = issuer_inputs
    _, identities, inputs = build_issuer_identity_inputs(root=root, config=config, expected_config_sha256=file_sha256(config))
    manifest: dict[str, object] = {"inputs": inputs}
    good = {"security_id": "cik:0000798354", "ticker": "FISV", "title": "$FISV reports earnings",
        "feature_available_at_utc": "2023-06-07T13:29:59.999999999Z"}
    validate_issuer_event_scope(pd.DataFrame([good]), identities, manifest)
    for override in ({"feature_available_at_utc": "2023-06-07T13:30:00Z"},
        {"feature_available_at_utc": "2019-07-08T23:59:59Z"}, {"feature_available_at_utc": None},
        {"security_id": "cusip:337738108"}, {"ticker": "FI"}):
        with pytest.raises(DataReadinessError, match="outside"):
            validate_issuer_event_scope(pd.DataFrame([{**good, **override}]), identities, manifest)


def test_document_replay_pin_is_required(tmp_path: Path) -> None:
    (tmp_path / "archive").mkdir()
    (tmp_path / "inventory").write_text("synthetic test-only inventory", encoding="utf-8")
    namespace = "market_predictor.sources.official_documents."
    with patch(namespace + "load_official_document_inventory"), \
        patch(namespace + "verify_official_document_collection", return_value={"status": "collected_unreviewed"}):
        with pytest.raises(DataReadinessError, match="report pin"):
            _document_bodies(tmp_path, "inventory", "archive", "f" * 64, set())


def test_identity_replay_reloads_the_pinned_correction_policy(issuer_inputs) -> None:  # type: ignore[no-untyped-def]
    root, config, _ = issuer_inputs
    result = publish_issuer_identity_authority(root=root, config=config,
        expected_config_sha256=file_sha256(config), output_directory=root / "policy-replay")
    labels, label_manifest = load_canonical_artifact(Path(result["business_labels"]), allow_research=True)
    identities, identity_manifest = load_canonical_artifact(Path(result["security_identities"]), allow_research=True)
    verify_issuer_identity_inputs(labels, identities, label_manifest, identity_manifest)
    correction_path = root / tomllib.loads(config.read_text(encoding="utf-8"))["correction_policy"]
    correction_path.write_bytes(correction_path.read_bytes() + b"\n# changed after publication\n")
    with pytest.raises(DataReadinessError, match="independent reviewed policy pin"):
        verify_issuer_identity_inputs(labels, identities, label_manifest, identity_manifest)


@pytest.mark.parametrize("lease_available", [True, False])
@pytest.mark.parametrize("relative_paths", [True, False])
def test_identity_command_preserves_flags_and_acquires_lease_before_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, lease_available: bool,
    relative_paths: bool,
) -> None:
    from market_predictor.commands import issuer_identity_publication as command

    root, other_cwd = tmp_path / "repository", tmp_path / "other-cwd"
    root.mkdir()
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)
    config, output = root / "issuer.toml", root / "authority"
    argv = ["issuer_identity_publication", "--root", str(root),
        "--config", str(config.relative_to(root) if relative_paths else config),
        "--expected-config-sha256", "a" * 64,
        "--out-dir", str(output.relative_to(root) if relative_paths else output)]
    result = {"security_identities": str(output / "security_identities.parquet")}
    with patch("sys.argv", argv), patch.object(command, "heavy_job_lease") as lease, \
        patch.object(command, "publish_issuer_identity_authority") as publish:
        context = lease.return_value

        def publish_inside_lease(**kwargs: object) -> dict[str, str]:
            context.__enter__.assert_called_once_with()
            context.__exit__.assert_not_called()
            return result

        publish.side_effect = publish_inside_lease
        if lease_available:
            command.main()
            publish.assert_called_once_with(root=root, config=config,
                expected_config_sha256="a" * 64, output_directory=output)
            context.__exit__.assert_called_once_with(None, None, None)
            assert json.loads(capsys.readouterr().out) == result
        else:
            context.__enter__.side_effect = DataReadinessError("lease unavailable")
            with pytest.raises(DataReadinessError, match="lease unavailable"):
                command.main()
            publish.assert_not_called()
            assert capsys.readouterr().out == ""
        lease.assert_called_once_with("publish-issuer-identity-authority", runtime_dir=root / "data/runtime")
