"""Legacy query identity proofs over IDs minted by the real legacy and cohort minting functions."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.universe import legacy_query_identity as legacy
from market_predictor.universe.issuer_news_identity import (
    BRIDGE_COLUMNS,
    _provenance,
    map_news_coverage,
    map_news_relations,
)
from market_predictor.universe.legacy_query_identity import (
    LEGACY_STATUS,
    PROOF_COLUMNS,
    build_legacy_query_proofs,
    coverage_pass,
    map_legacy_query_coverage,
    map_legacy_query_relations,
    relations_pass,
    translation_index,
)
from market_predictor.universe.sp500.membership_authority import _historical_security_id
from market_predictor.universe.sp500.membership_history import (
    IndexChange,
    IndexChangeSource,
    _security_identity_for_interval,
)

PIN = "a" * 64
AUTHORITY = ("security_id", "ticker", "effective_from_utc", "effective_to_utc", "available_at_utc")
TRANSITIONS = ("id", "old_symbol", "new_symbol", "old_cusip", "new_cusip", "identity_continuity", "effective_date",
               "process_date")
NEW_IR, TRANE = "cik:0001699150", "cik:0001466258"


def _t(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="America/New_York").tz_convert("UTC")


def _row(security: str, ticker: str, start: str, end: str | None = None) -> dict[str, Any]:
    return {"security_id": security, "ticker": ticker, "effective_from_utc": _t(start),
            "effective_to_utc": _t(end) if end else pd.NaT, "available_at_utc": _t(start)}


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=AUTHORITY)
    for column in AUTHORITY[2:]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _change(action: str, ticker: str, company: str, day: str, published: str = "2019-01-02") -> IndexChange:
    source = IndexChangeSource(source_url=f"https://press.spglobal.com/{ticker}-{action}",
                               source_published_date=date.fromisoformat(published), source_sha256="b" * 64)
    return IndexChange(effective_at_utc=_t(day).to_pydatetime(), action=action, ticker=ticker, company=company,
                       sector="Industrials", source_url=source.source_url, source_published_date=source.source_published_date,
                       source_sha256=source.source_sha256, supporting_sources=(source,))


def _legacy_id(ticker: str, company: str, start: str, end: str | None = None) -> str:
    return _security_identity_for_interval(ticker=ticker, company=company, effective_from=_t(start),
                                           effective_to=_t(end) if end else None, current=pd.DataFrame(), aliases=[])


def _target_id(ticker: str, company: str) -> str:
    return _historical_security_id(_change("deletion", ticker, company, "2021-01-04"))


def _transition(identifier: str, day: str, old: str, new: str, cusip: str, *, identity: bool = True,
                new_cusip: str | None = None) -> dict[str, Any]:
    return {"id": identifier, "old_symbol": old, "new_symbol": new, "old_cusip": cusip, "new_cusip": new_cusip or cusip,
            "identity_continuity": identity, "effective_date": day, "process_date": day}


def _build(legacy_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], *,
           changes: tuple[IndexChange, ...] = (), transitions: tuple[dict[str, Any], ...] = (),
           bridge: pd.DataFrame | None = None, corrected: tuple[str, ...] = (), extra: tuple[pd.DataFrame, ...] = (),
           evidence: str = PIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_legacy_query_proofs(
        [_frame(*legacy_rows), *extra], _frame(*target_rows), changes, pd.DataFrame(list(transitions), columns=TRANSITIONS),
        pd.DataFrame(columns=BRIDGE_COLUMNS) if bridge is None else bridge, corrected_security_ids=corrected,
        target_cutoff_utc=_t("2026-07-08"), evidence_sha256=evidence)


def _reasons(rejections: pd.DataFrame) -> dict[str, str]:
    return dict(zip(rejections.source_security_id + "|" + rejections.ticker, rejections.reason, strict=True))


def test_company_hash_proof_reproduces_both_real_mints_and_rejects_tampering() -> None:
    company, events = "Old Holdings Inc", (_change("addition", "OLD", "Old Holdings Inc", "2018-01-02", "2017-12-20"),
                                            _change("deletion", "OLD", "Old Holdings Inc", "2021-01-04", "2020-12-18"))
    source, target = _legacy_id("OLD", company, "2019-07-09", "2021-01-04"), _target_id("OLD", company)
    proofs, rejections = _build([_row(source, "OLD", "2019-07-09", "2021-01-04")],
                                [_row(target, "OLD", "2018-05-29", "2021-01-04")], changes=events)
    assert rejections.empty and len(proofs) == 1
    proof = proofs.iloc[0]
    assert (proof.target_security_id, proof.proof_kind, proof.evidence_complete_date) == (
        target, "company_ticker_hash_reproduced", "2017-12-20")
    assert (proof.effective_from_utc, proof.available_at_utc) == (_t("2019-07-09"), _t("2019-07-09"))
    evidence = json.loads(proof.evidence_json)
    assert evidence["company"] == "old holdings inc" and len(evidence["events"]) == 2
    assert {source["source_sha256"] for event in evidence["events"] for source in event["sources"]} == {"b" * 64}
    for name, spell in (("company", _legacy_id("OLD", "Old Holding Inc", "2019-07-09", "2021-01-04")),
                        ("start", _legacy_id("OLD", company, "2019-07-10", "2021-01-04")),
                        ("ticker", _legacy_id("OLDX", company, "2019-07-09", "2021-01-04"))):
        _, rejected = _build([_row(spell, "OLD", "2019-07-09", "2021-01-04")],
                             [_row(target, "OLD", "2018-05-29", "2021-01-04")], changes=events)
        assert rejected.reason.tolist() == ["no_candidate"], name


def test_a_company_name_outside_the_pinned_events_is_never_evidence() -> None:
    source = _legacy_id("OLD", "Old Holdings Inc", "2019-07-09", "2021-01-04")
    pinned_elsewhere = _change("deletion", "OLD", "Old Holdings Inc", "2021-01-04")
    target = _historical_security_id(pinned_elsewhere)
    _, rejections = _build([_row(source, "OLD", "2019-07-09", "2021-01-04")],
                           [_row(target, "OLD", "2018-05-29", "2021-01-04")],
                           changes=(_change("deletion", "OLD", "Old Holdings Corporation", "2021-01-04"),))
    assert rejections.reason.tolist() == ["no_candidate"]


def test_spell_events_bind_addition_and_deletion_names_to_identical_boundaries() -> None:
    added, removed = (_change("addition", "OGN", "Organon", "2021-06-03", "2021-05-27"),
                      _change("deletion", "OGN", "Organon & Co", "2023-10-18", "2023-10-13"))
    source, target = _legacy_id("OGN", "Organon", "2021-06-03", "2023-10-18"), _historical_security_id(removed)
    proofs, rejections = _build([_row(source, "OGN", "2021-06-03", "2023-10-18")],
                                [_row(target, "OGN", "2021-06-03", "2023-10-18")], changes=(added, removed))
    assert rejections.empty and proofs.proof_kind.tolist() == ["sp500_spell_events_reproduced"]
    assert proofs.evidence_complete_date.item() == "2023-10-13"
    assert set(json.loads(proofs.evidence_json.item())) == {"addition", "deletion"}
    _, shifted = _build([_row(source, "OGN", "2021-06-03", "2023-10-18")],
                        [_row(target, "OGN", "2021-06-04", "2023-10-18")], changes=(added, removed))
    assert shifted.reason.tolist() == ["no_candidate"]


def test_cik_equality_requires_the_exact_query_ticker() -> None:
    proofs, rejections = _build([_row("cik:0001067983:ticker:BRK-B", "BRK-B", "2019-07-09")],
                                [_row("cik:0001067983", "BRK-B", "2018-05-29")])
    assert rejections.empty and (proofs.target_security_id.item(), proofs.proof_kind.item()) == ("cik:0001067983", "cik_equal")
    _, differs = _build([_row("cik:0001067983:ticker:BRK-B", "BRK-B", "2019-07-09")],
                        [_row("cik:0001067983", "BRK-A", "2018-05-29")])
    assert differs.reason.tolist() == ["cik_equal_ticker_differs"]
    _, missing = _build([_row("cik:0001067983:ticker:BRK-B", "BRK-B", "2019-07-09")], [_row("cik:0000000001", "A", "2018-05-29")])
    assert missing.reason.tolist() == ["no_candidate"]


def test_latest_ticker_target_authority_never_hands_trane_ir_news_to_ingersoll_rand() -> None:
    # The target authority carries both securities' latest tickers back to 2018, as the real one does.
    targets = [_row(NEW_IR, "IR", "2018-05-29"), _row(TRANE, "TT", "2018-05-29")]
    chains = [_row("cusip:G8994E103", "IR", "2019-07-09", "2020-03-02"), _row("cusip:G8994E103", "TT", "2020-03-02"),
              _row("cusip:45687V106", "IR", "2020-03-03")]
    proofs, rejections = _build(chains, targets,
                                transitions=(_transition("ir-tt", "2020-03-02", "IR", "TT", "G8994E103"),))
    assert rejections.empty
    rows = {(row.source_security_id, row.ticker): row for row in proofs.itertuples()}
    assert rows[("cusip:G8994E103", "IR")].target_security_id == TRANE
    assert rows[("cusip:G8994E103", "TT")].target_security_id == TRANE
    assert rows[("cusip:45687V106", "IR")].target_security_id == NEW_IR
    assert rows[("cusip:45687V106", "IR")].effective_from_utc == _t("2020-03-03")
    assert set(proofs.proof_kind) == {"cusip_chain_end_ticker_match"}
    evidence = json.loads(rows[("cusip:G8994E103", "IR")].evidence_json)
    assert [item["id"] for item in evidence["transitions"]] == ["ir-tt"] and evidence["matched_ticker"] == "TT"
    assert rows[("cusip:G8994E103", "IR")].evidence_complete_date == "2026-07-08"
    _, unexplained = _build(chains, targets)
    assert _reasons(unexplained) == {"cusip:G8994E103|IR": "legacy_chain_contradicts_transitions",
                                     "cusip:G8994E103|TT": "legacy_chain_contradicts_transitions"}


def test_a_chain_its_own_transitions_contradict_is_rejected_and_merger_rows_are_not_contradictions() -> None:
    chain = [_row("cusip:337738108", "FI", "2019-07-09", "2025-11-11"), _row("cusip:337738108", "FISV", "2025-11-11")]
    target = [_row("cik:0000798354", "FISV", "2018-05-29")]
    own = (_transition("fisv-fi", "2023-06-07", "FISV", "FI", "337738108"),
           _transition("fi-fisv", "2025-11-11", "FI", "FISV", "337738108"))
    _, rejections = _build(chain, target, transitions=own)
    assert set(rejections.reason) == {"legacy_chain_contradicts_transitions"} and len(rejections) == 2
    assert json.loads(rejections.detail_json.iloc[0])["transition_id"] == "fisv-fi"
    merger = _transition("merger", "2021-10-01", "XEC", "FI", "171798101", identity=False, new_cusip="337738108")
    proofs, clean = _build(chain, target, transitions=(own[1], merger))
    assert clean.empty and proofs.target_security_id.eq("cik:0000798354").all()
    _, corrected = _build(chain, target, transitions=(own[1], merger), corrected=("cik:0000798354",))
    assert set(corrected.reason) == {"corrected_security_uses_corrections_archive_only"}
    _, unexplained = _build(chain, target)
    assert set(unexplained.reason) == {"legacy_chain_contradicts_transitions"}


def test_minting_parser_when_issued_filter_does_not_hide_a_contradiction() -> None:
    # symbol_changes_from_transitions drops old tickers ending in V; verification must still see them.
    chain = [_row("cusip:000000001", "ABC", "2019-07-09")]
    _, rejections = _build(chain, [_row("cik:0000000001", "ABC", "2018-05-29")],
                           transitions=(_transition("abcv", "2021-01-04", "ABCV", "ABC", "000000001"),))
    assert rejections.reason.tolist() == ["legacy_chain_contradicts_transitions"]


def test_non_continuous_transition_stays_split_and_entry_renames_are_consistent() -> None:
    chains = [_row("cusip:339041105", "FLT", "2019-07-09", "2024-03-25"), _row("cusip:219948106", "CPAY", "2024-03-25")]
    transitions = (_transition("flt-cpay", "2024-03-25", "FLT", "CPAY", "339041105", identity=False, new_cusip="219948106"),
                   _transition("entry", "2024-03-25", "CPAYW", "CPAY", "219948106"))
    proofs, rejections = _build(chains, [_row("cik:0001175454", "CPAY", "2018-05-29")], transitions=transitions)
    assert _reasons(rejections) == {"cusip:339041105|FLT": "no_candidate"}
    assert (proofs.source_security_id.item(), proofs.effective_from_utc.item()) == ("cusip:219948106", _t("2024-03-25"))


def test_a_rejoined_targets_earlier_row_end_is_not_its_final_instant() -> None:
    # The target left and rejoined; its earlier row carries the latest ticker back, so that end proves nothing.
    chain = [_row("cusip:000000005", "T", "2019-07-09", "2021-01-04")]
    rejoined = [_row("cik:0000000005", "T", "2018-05-29", "2021-01-04"), _row("cik:0000000005", "T", "2022-01-03")]
    _, rejections = _build(chain, rejoined)
    assert rejections.reason.tolist() == ["no_candidate"]
    proofs, _ = _build([_row("cusip:000000005", "T", "2019-07-09")], rejoined)
    assert proofs.target_security_id.eq("cik:0000000005").all() and len(proofs) == 2


def test_ticker_handover_is_ambiguous_and_reuse_at_other_times_is_not() -> None:
    chain = [_row("cusip:000000002", "T", "2019-07-09")]
    handover = [_row("cik:0000000001", "T", "2018-05-29", "2020-01-02"), _row("cik:0000000002", "T", "2020-01-02")]
    deleted = (_change("deletion", "T", "First T Inc", "2020-01-02", "2019-12-20"),)
    _, ambiguous = _build(chain, handover, changes=deleted)
    assert ambiguous.reason.tolist() == ["ambiguous_candidates"]
    assert json.loads(ambiguous.detail_json.item())["targets"] == ["cik:0000000001", "cik:0000000002"]
    closed = [_row("cusip:000000002", "T", "2019-07-09", "2021-01-04")]
    reused = [_row("cik:0000000001", "T", "2018-05-29", "2021-01-04"), _row("cik:0000000002", "T", "2022-01-03")]
    closing = (_change("deletion", "T", "First T Inc", "2021-01-04", "2020-12-18"),)
    proofs, rejections = _build(closed, reused, changes=closing)
    assert rejections.empty and proofs.target_security_id.tolist() == ["cik:0000000001"]
    evidence = json.loads(proofs.evidence_json.item())
    assert evidence["closing_deletion"][0]["effective_at_utc"] == _t("2021-01-04").isoformat()
    assert proofs.evidence_complete_date.item() == "2021-01-04"
    # A closed row's end is not a true-ticker instant without a pinned deletion documenting it.
    _, undocumented = _build(closed, reused)
    assert undocumented.reason.tolist() == ["no_candidate"]


def test_intersections_split_at_target_rows_and_outside_spells_are_rejected() -> None:
    chain = [_row("cusip:000000003", "AAA", "2019-07-09", "2019-12-02"), _row("cusip:000000003", "BBB", "2019-12-02")]
    target = [_row("cik:0000000003", "BBB", "2020-01-02", "2021-01-04"), _row("cik:0000000003", "BBB", "2022-01-03")]
    transitions = (_transition("aaa-bbb", "2019-12-02", "AAA", "BBB", "000000003"),)
    proofs, rejections = _build(chain, target, transitions=transitions)
    assert _reasons(rejections) == {"cusip:000000003|AAA": "no_membership_intersection"}
    assert proofs.effective_from_utc.tolist() == [_t("2020-01-02"), _t("2022-01-03")]
    assert proofs.effective_to_utc.iloc[0] == _t("2021-01-04") and pd.isna(proofs.effective_to_utc.iloc[1])
    assert proofs.legacy_spell_from_utc.eq(_t("2019-12-02")).all()


def test_scope_namespaces_spells_and_conservation() -> None:
    historical = _legacy_id("OLD", "Old Co", "2019-07-09", "2020-01-02")
    rows = [_row(historical, "OLD", "2019-07-09", "2020-01-02"), _row(historical, "OLD", "2021-01-04", "2022-01-03"),
            _row("transition:abc", "NEW", "2019-07-09"), _row("cik:0000000009:ticker:BRG", "BRG", "2019-07-09"),
            _row("cik:0000000008", "EQL", "2019-07-09")]
    bridge = pd.DataFrame([{"source_security_id": "cik:0000000009:ticker:BRG", "ticker": "BRG",
                            "target_security_id": "cik:0000000009", "effective_from_utc": _t("2019-07-09"),
                            "effective_to_utc": pd.NaT, "available_at_utc": _t("2019-07-09"), "bridge_row_sha256": PIN}])
    proofs, rejections = _build(rows, [_row("cik:0000000008", "EQL", "2018-05-29"), _row("cik:0000000009", "BRG", "2018-05-29")],
                                bridge=bridge)
    assert proofs.empty and list(proofs.columns) == list(PROOF_COLUMNS)
    assert _reasons(rejections) == {f"{historical}|OLD": "multiple_legacy_spells", "transition:abc|NEW": "unsupported_identity_namespace"}
    assert len(rejections) == 3


@pytest.mark.parametrize("claim", ["bridge", "identity_equal", "proof"])
def test_a_proof_overlapping_any_other_claim_on_its_target_fails(claim: str) -> None:
    rows = [_row("cik:0000000001:ticker:AAA", "AAA", "2019-07-09")]
    bridge = None
    if claim == "bridge":
        bridge = pd.DataFrame([{"source_security_id": "cik:0000000002:ticker:ZZZ", "ticker": "ZZZ",
                                "target_security_id": "cik:0000000001", "effective_from_utc": _t("2020-01-02"),
                                "effective_to_utc": pd.NaT, "available_at_utc": _t("2020-01-02"), "bridge_row_sha256": PIN}])
    elif claim == "identity_equal":
        rows.append(_row("cik:0000000001", "AAA", "2021-01-04"))
    else:
        rows = [_row("cusip:000000011", "AAA", "2019-07-09"), _row("cusip:000000012", "AAA", "2021-01-04")]
    targets = [_row("cik:0000000001", "AAA", "2018-05-29")]
    with pytest.raises(DataReadinessError, match="overlaps another claim"):
        _build(rows, targets, bridge=bridge)


def test_legacy_files_must_agree_on_shared_identities() -> None:
    source = _row("cik:0001067983:ticker:BRK-B", "BRK-B", "2019-07-09")
    with pytest.raises(DataReadinessError, match="disagree"):
        _build([source], [_row("cik:0001067983", "BRK-B", "2018-05-29")],
               extra=(_frame({**source, "effective_from_utc": _t("2019-07-10"), "available_at_utc": _t("2019-07-10")}),))
    proofs, _ = _build([source], [_row("cik:0001067983", "BRK-B", "2018-05-29")], extra=(_frame(source),))
    assert len(proofs) == 1


def test_proof_rows_are_deterministic_and_bound_to_their_evidence_pins() -> None:
    args = ([_row("cik:0001067983:ticker:BRK-B", "BRK-B", "2019-07-09")], [_row("cik:0001067983", "BRK-B", "2018-05-29")])
    first, second, other = _build(*args)[0], _build(*args)[0], _build(*args, evidence="c" * 64)[0]
    pd.testing.assert_frame_equal(first, second)
    assert first.proof_row_sha256.item() != other.proof_row_sha256.item()


def test_transition_evidence_is_strict() -> None:
    chain, target = [_row("cusip:000000004", "AAA", "2019-07-09")], [_row("cik:0000000004", "AAA", "2018-05-29")]
    with pytest.raises(DataReadinessError, match="changes its CUSIP"):
        _build(chain, target, transitions=(_transition("x", "2020-01-02", "OLD", "AAA", "000000004", new_cusip="000000005"),))
    with pytest.raises(DataReadinessError, match="no effective date"):
        _build(chain, target, transitions=({**_transition("x", "2020-01-02", "OLD", "AAA", "000000004"),
                                            "effective_date": None, "process_date": None},))


# Translation: the two passes reproduce the protected CIK functions and add legacy proofs as a second pass.
OLD, NEW = "cik:0000320193:ticker:AAPL", "cik:0000320193"


def _day(day: int) -> pd.Timestamp:
    return pd.Timestamp(f"2020-01-{day:02d}T00:00:00Z")


def _cik_bridge() -> pd.DataFrame:
    return pd.DataFrame([{"source_security_id": OLD, "ticker": "AAPL", "target_security_id": NEW,
                          "effective_from_utc": _day(3), "effective_to_utc": _day(10), "available_at_utc": _day(5),
                          "bridge_row_sha256": "d" * 64},
                         {"source_security_id": OLD, "ticker": "AAPL", "target_security_id": NEW,
                          "effective_from_utc": _day(12), "effective_to_utc": _day(15), "available_at_utc": _day(12),
                          "bridge_row_sha256": "e" * 64}])


def _relations(security: str = OLD, ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{"relation_id": f"relation-{security}-{day}", "target_security_id": security, "target_ticker": ticker,
                          "event_feature_available_at_utc": _day(day), "identity_available_at_utc": _day(1),
                          "feature_available_at_utc": _day(20), "sentiment_numeric": 0.25} for day in (1, 4, 6, 13, 16)])


def _ledger(security: str = OLD, ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{"chunk_id": f"chunk-{security}", "security_id": security, "ticker": ticker,
                          "requested_start_utc": _day(1), "requested_end_utc": _day(20), "status": "observed"}])


def _proofs(security: str = "sp500-historical:abc", ticker: str = "OLD") -> pd.DataFrame:
    frame = pd.DataFrame([{"source_security_id": security, "ticker": ticker, "target_security_id": "cik:0000000077",
                           "effective_from_utc": _day(2), "effective_to_utc": _day(8), "available_at_utc": _day(4),
                           "legacy_spell_from_utc": _day(1), "legacy_spell_to_utc": _day(8),
                           "proof_kind": "company_ticker_hash_reproduced", "evidence_json": "{}",
                           "evidence_complete_date": "2019-12-31", "proof_row_sha256": "f" * 64}], columns=PROOF_COLUMNS)
    return frame


def test_generic_passes_reproduce_both_protected_cik_functions_exactly() -> None:
    # A synthetic unit differential; the real-bridge run over every ledger and record is recorded in the handoff.
    bridge = _cik_bridge()
    index = translation_index(bridge, source_column="source_security_id", digest_column="bridge_row_sha256")
    frame = pd.concat([_relations(), _relations("cik:0000000009:ticker:ZZZ", "ZZZ")], ignore_index=True)
    frame.loc[3, "identity_available_at_utc"] = _day(14)  # Proven identity arrives after the day-13 event: skipped.
    relations = _provenance(frame.copy(), ("relation_id", "target_security_id", "target_ticker",
                                           "identity_available_at_utc", "feature_available_at_utc"))
    relations_pass(relations, index, [True] * len(relations), status="mapped", digest_key="bridge_row_sha256",
                   digest_column="identity_bridge_row_sha256")
    pd.testing.assert_frame_equal(relations, map_news_relations(frame, bridge))
    assert relations.identity_translation_status.tolist() == ["unmapped", "unmapped", "mapped", *["unmapped"] * 7]
    columns = ("chunk_id", "security_id", "ticker", "requested_start_utc", "requested_end_utc")
    ledgers = pd.concat([_ledger(), _ledger("cik:0000000009:ticker:ZZZ", "ZZZ")], ignore_index=True)
    ledger = _provenance(ledgers.copy(), columns)
    ledger["identity_available_at_utc"] = pd.Series(pd.NaT, index=ledger.index, dtype="datetime64[ns, UTC]")
    result = coverage_pass(ledger, index, [True, True], status="mapped", digest_key="bridge_row_sha256",
                           digest_column="identity_bridge_row_sha256")
    pd.testing.assert_frame_equal(result, map_news_coverage(ledgers, bridge))
    assert result.identity_translation_status.tolist() == ["unmapped", "mapped", "unmapped", "mapped", "unmapped", "unmapped"]


def test_legacy_relations_translate_only_cik_unmapped_rows_at_event_time() -> None:
    frame = pd.concat([_relations(), _relations("sp500-historical:abc", "OLD")], ignore_index=True)
    first = map_news_relations(frame, _cik_bridge())
    result = map_legacy_query_relations(first, _proofs())
    legacy_rows = result.identity_translation_status.eq(LEGACY_STATUS)
    assert result.loc[legacy_rows, "identity_original_target_security_id"].eq("sp500-historical:abc").all()
    # Event day 1 precedes the proof, day 4 is at its availability, day 6 inside it, days 13 and 16 after it.
    assert result.loc[5:, "identity_translation_status"].tolist() == ["unmapped", LEGACY_STATUS, LEGACY_STATUS,
                                                                     "unmapped", "unmapped"]
    assert result.loc[legacy_rows, "target_security_id"].eq("cik:0000000077").all()
    assert result.loc[legacy_rows, "identity_legacy_proof_kind"].eq("company_ticker_hash_reproduced").all()
    assert result.loc[legacy_rows, "identity_legacy_proof_row_sha256"].eq("f" * 64).all()
    assert result.loc[legacy_rows, "identity_available_at_utc"].eq(_day(4)).all()
    assert result.loc[legacy_rows, "identity_bridge_row_sha256"].eq("").all()
    assert result.loc[~legacy_rows, "identity_legacy_proof_kind"].eq("").all()
    unchanged = [column for column in first.columns if column not in {
        "target_security_id", "relation_id", "identity_translation_status", "identity_available_at_utc",
        "feature_available_at_utc"}]
    pd.testing.assert_frame_equal(result[unchanged], first[unchanged])
    pd.testing.assert_frame_equal(result.loc[:4, first.columns], first.loc[:4])
    assert result.relation_id.is_unique and set(result.loc[legacy_rows, "relation_id"]).isdisjoint(first.relation_id)
    with pytest.raises(DataReadinessError, match="legacy-untranslated"):
        map_legacy_query_relations(result, _proofs())
    with pytest.raises(DataReadinessError, match="missing columns"):
        map_legacy_query_relations(frame, _proofs())


def test_legacy_coverage_splits_only_unmapped_segments_without_inventing_time() -> None:
    frame = pd.concat([_ledger(), _ledger("sp500-historical:abc", "OLD")], ignore_index=True)
    first = map_news_coverage(frame, _cik_bridge())
    result = map_legacy_query_coverage(first, _proofs())
    pd.testing.assert_frame_equal(result.loc[result.identity_original_security_id.eq(OLD), first.columns].reset_index(drop=True),
                                  first.loc[first.identity_original_security_id.eq(OLD)].reset_index(drop=True))
    proven = result.loc[result.identity_original_security_id.eq("sp500-historical:abc")]
    assert proven.requested_start_utc.tolist() == [_day(1), _day(4), _day(8)]
    assert proven.identity_translation_status.tolist() == ["unmapped", LEGACY_STATUS, "unmapped"]
    assert proven.security_id.tolist() == ["sp500-historical:abc", "cik:0000000077", "sp500-historical:abc"]
    assert (proven.requested_end_utc - proven.requested_start_utc).sum() == _day(20) - _day(1)
    assert result.chunk_id.is_unique and proven.identity_available_at_utc.iloc[1] == _day(4)
    assert proven.identity_legacy_proof_kind.tolist() == ["", "company_ticker_hash_reproduced", ""]
    with pytest.raises(DataReadinessError, match="legacy-untranslated"):
        map_legacy_query_coverage(result, _proofs())


def test_proof_frames_are_validated_before_translation() -> None:
    first = map_news_relations(_relations("sp500-historical:abc", "OLD"), _cik_bridge())
    for change in ({"proof_kind": "ticker_guess"}, {"target_security_id": "sp500-historical:abc"}):
        with pytest.raises(DataReadinessError):
            map_legacy_query_relations(first, _proofs().assign(**change))
    with pytest.raises(DataReadinessError, match="overlapping"):
        map_legacy_query_relations(first, pd.concat([_proofs(), _proofs().assign(proof_row_sha256="0" * 64)]))
    with pytest.raises(DataReadinessError, match="CIK-pass statuses"):
        map_legacy_query_relations(first.assign(identity_translation_status="legacy"), _proofs())
    assert legacy.AVAILABILITY_BASIS == "retrospective_membership_effective_proxy"
