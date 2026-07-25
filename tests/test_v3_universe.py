from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.universe import (
    build_point_in_time_sp500_universe,
    load_reviewed_security_transitions,
    merge_security_transition_evidence,
    parse_sp500_changes,
    symbol_changes_from_transitions,
)


class V3PointInTimeUniverseTests(unittest.TestCase):
    def test_reviewed_transition_ledger_overrides_unapproved_provider_merger(self) -> None:
        reviewed = load_reviewed_security_transitions(Path("configs/sp500_security_transition_review.csv"))
        provider = pd.DataFrame(
            {
                "id": ["provider-para", "provider-cbs-wrong-date", "provider-unreviewed"],
                "effective_date": ["2025-08-07", "2020-02-13", "2025-01-02"],
                "old_symbol": ["PARA", "CBS", "AAA"],
                "new_symbol": ["PSKY", "VIAC", "BBB"],
                "transition_type": ["stock_merger", "name_change", "stock_merger"],
                "identity_continuity": [False, True, False],
                "membership_continuity": [False, True, False],
            }
        )

        merged = merge_security_transition_evidence(provider, reviewed)
        changes = symbol_changes_from_transitions(merged)

        pairs = {(change.old_ticker, change.new_ticker) for change in changes}
        self.assertIn(("PARA", "PSKY"), pairs)
        self.assertNotIn(("AAA", "BBB"), pairs)
        para = next(change for change in changes if change.old_ticker == "PARA")
        self.assertEqual(para.source_url.split("/")[2], "www.sec.gov")
        self.assertEqual(para.old_security_id, "cik:0000813828")
        self.assertEqual(para.new_security_id, "cik:0002041610")
        cbs = [change for change in changes if change.old_ticker == "CBS"]
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].effective_at_utc.date(), date(2019, 12, 5))

    def test_transition_overlay_does_not_stringify_missing_security_ids_as_nan(self) -> None:
        reviewed = load_reviewed_security_transitions(Path("configs/sp500_security_transition_review.csv"))
        provider = pd.DataFrame(
            {
                "id": ["provider-rename"],
                "effective_date": ["2024-01-02"],
                "old_symbol": ["OLD"],
                "new_symbol": ["NEW"],
                "old_cusip": ["111111111"],
                "new_cusip": ["111111111"],
                "transition_type": ["name_change"],
                "identity_continuity": [True],
                "membership_continuity": [True],
            }
        )

        changes = symbol_changes_from_transitions(merge_security_transition_evidence(provider, reviewed))
        rename = next(change for change in changes if change.old_ticker == "OLD")

        self.assertEqual(rename.old_security_id, "cusip:111111111")
        self.assertEqual(rename.new_security_id, "cusip:111111111")

    def test_parses_only_sp500_rows_from_official_table(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html(),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        self.assertEqual([(item.action, item.ticker) for item in changes], [("addition", "NEW"), ("deletion", "OLD")])
        self.assertEqual(changes[0].effective_at_utc.isoformat(), "2026-01-05T05:00:00+00:00")

    def test_structured_table_forward_fills_effective_date(self) -> None:
        html = _announcement_html().replace(
            "<tr><td>January 5, 2026</td><td>S&amp;P 500</td><td>Deletion</td>",
            "<tr><td></td><td>S&amp;P 500</td><td>Deletion</td>",
        )

        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2026-01-01-forward-filled-date",
            published_date=date(2026, 1, 1),
        )

        self.assertEqual([(item.action, item.ticker) for item in changes], [("addition", "NEW"), ("deletion", "OLD")])
        self.assertEqual({item.effective_at_utc.isoformat() for item in changes}, {"2026-01-05T05:00:00+00:00"})

    def test_structured_composite_ticker_expands_to_tradable_share_classes(self) -> None:
        html = _announcement_html().replace("OLD", "UA/UAA")
        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2022-06-03-under-armour",
            published_date=date(2022, 6, 3),
        )

        self.assertEqual(
            [(item.action, item.ticker) for item in changes],
            [("addition", "NEW"), ("deletion", "UA"), ("deletion", "UAA")],
        )

    def test_parses_legacy_official_table_using_same_release_tickers(self) -> None:
        changes = parse_sp500_changes(
            _legacy_announcement_html(),
            source_url="https://press.spglobal.com/2019-07-09-example",
            published_date=date(2019, 7, 9),
        )

        self.assertEqual(
            [(item.action, item.ticker, item.company) for item in changes],
            [
                ("addition", "TMUS", "T-Mobile US"),
                ("deletion", "RHT", "Red Hat"),
            ],
        )
        self.assertEqual(changes[0].effective_at_utc.isoformat(), "2019-07-15T04:00:00+00:00")

    def test_parses_sp500_section_embedded_after_another_index(self) -> None:
        html = """
        <html><body><div class="wd_news_body">
          <p>
            T-Mobile US Inc (NASD: TMUS) will replace Red Hat Inc. (NYSE: RHT)
            in the S&amp;P 500.
          </p>
          <table>
            <tr><td>S&amp;P SMALLCAP 600 INDEX – July 12, 2019</td></tr>
            <tr><td></td><td>COMPANY</td><td>GICS ECONOMIC SECTOR</td></tr>
            <tr><td>ADDED</td><td>Unrelated Company</td><td>Industrials</td></tr>
            <tr><td></td></tr>
            <tr><td>S&amp;P 500 INDEX – July 15, 2019</td></tr>
            <tr><td></td><td>COMPANY</td><td>GICS ECONOMIC SECTOR</td></tr>
            <tr><td>ADDED</td><td>T-Mobile US</td><td>Communication Services</td></tr>
            <tr><td>DELETED</td><td>Red Hat</td><td>Information Technology</td></tr>
          </table>
        </div></body></html>
        """

        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2019-07-09-multi-index",
            published_date=date(2019, 7, 9),
        )

        self.assertEqual([(item.action, item.ticker) for item in changes], [("addition", "TMUS"), ("deletion", "RHT")])

    def test_legacy_table_carries_action_across_blank_action_cells(self) -> None:
        html = """
        <html><body><div class="wd_news_body">
          <p>
            T-Mobile US Inc (NASD: TMUS) and Harley- Davidson Inc. (NYSE: HOG)
            will move to the S&amp;P 500, replacing Red Hat Inc. (NYSE: RHT).
          </p>
          <table>
            <tr><td>S&amp;P 500 INDEX – June 22, 2020</td></tr>
            <tr><td></td><td>COMPANY</td><td>GICS ECONOMIC SECTOR</td></tr>
            <tr><td>ADDED</td><td>T-Mobile US</td><td>Communication Services</td></tr>
            <tr><td></td><td>Harley-Davidson</td><td>Consumer Discretionary</td></tr>
            <tr><td>DELETED</td><td>Red Hat</td><td>Information Technology</td></tr>
          </table>
        </div></body></html>
        """

        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2020-06-12-carried-action",
            published_date=date(2020, 6, 12),
        )

        self.assertEqual(
            [(item.action, item.ticker) for item in changes],
            [("addition", "HOG"), ("addition", "TMUS"), ("deletion", "RHT")],
        )

    def test_tba_effective_date_is_deferred_without_inventing_membership_time(self) -> None:
        html = _announcement_html().replace("January 5, 2026", "TBA")
        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2026-01-01-tba",
            published_date=date(2026, 1, 1),
        )

        self.assertEqual(changes, [])

    def test_legacy_table_fails_when_company_cannot_be_bound_to_ticker(self) -> None:
        html = _legacy_announcement_html().replace("(NYSE: RHT)", "")
        with self.assertRaises(DataReadinessError):
            parse_sp500_changes(
                html,
                source_url="https://press.spglobal.com/2019-07-09-ambiguous",
                published_date=date(2019, 7, 9),
            )

    def test_legacy_ticker_binding_does_not_capture_a_nearby_company(self) -> None:
        html = _legacy_announcement_html().replace(
            "T-Mobile US Inc (NASD: TMUS) will replace Red Hat Inc. (NYSE: RHT)",
            (
                "T-Mobile US Inc (NASD: TMUS) will replace Red Hat in a transaction involving "
                "Nearby Corp. (NYSE: BAD). Red Hat Inc. (NYSE: RHT)"
            ),
        )

        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2019-07-09-nearby-company",
            published_date=date(2019, 7, 9),
        )

        deleted = next(item for item in changes if item.action == "deletion")
        self.assertEqual(deleted.ticker, "RHT")

    def test_legacy_ticker_binding_accepts_bounded_legal_name_variants(self) -> None:
        html = """
        <html><body><div class="wd_news_body">
          <p>
            CDW Corp. (NASD: CDW) and Gardner Denver Holdings Inc. (NYSE:GDI)
            will join the S&amp;P 500.
          </p>
          <table>
            <tr><td>S&amp;P 500 INDEX – March 3, 2020</td></tr>
            <tr><td></td><td>COMPANY</td><td>GICS ECONOMIC SECTOR</td></tr>
            <tr><td>ADDED</td><td>CDW Corp</td><td>Information Technology</td></tr>
            <tr><td></td><td>Gardner Denver</td><td>Industrials</td></tr>
          </table>
        </div></body></html>
        """

        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2020-02-27-legal-variants",
            published_date=date(2020, 2, 27),
        )

        self.assertEqual([(item.action, item.ticker) for item in changes], [("addition", "CDW"), ("addition", "GDI")])

    def test_reverses_changes_into_non_overlapping_intervals(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html(),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        current = pd.DataFrame(
            {
                "ticker": ["AAA", "NEW"],
                "company": ["Always", "New Company"],
                "sector": ["Industrials", "Information Technology"],
                "industry": ["Services", "Software"],
            }
        )
        universe, audit = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            start_date=date(2025, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )
        old = universe[universe["ticker"] == "OLD"].iloc[0]
        new = universe[universe["ticker"] == "NEW"].iloc[0]
        self.assertEqual(pd.Timestamp(old["effective_to_utc"]), pd.Timestamp("2026-01-05T05:00:00Z"))
        self.assertEqual(pd.Timestamp(new["effective_from_utc"]), pd.Timestamp("2026-01-05T05:00:00Z"))
        self.assertEqual(new["primary_benchmark"], "XLK")
        self.assertEqual(audit["current_tickers"], 2)
        self.assertEqual(audit["historical_tickers"], 3)

    def test_fails_on_transition_contradiction(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html().replace("<td>OLD</td>", "<td>AAA</td>"),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        current = pd.DataFrame({"ticker": ["AAA", "NEW"], "sector": ["Industrials", "Information Technology"]})
        with self.assertRaises(DataReadinessError):
            build_point_in_time_sp500_universe(
                current_snapshot=current,
                changes=changes,
                start_date=date(2025, 7, 1),
                cutoff_date=date(2026, 7, 1),
                anchor_source="anchor.csv",
            )

    def test_symbol_change_preserves_membership_continuity(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html().replace("NEW", "OLD"),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        changes = [item for item in changes if item.action == "addition"]
        aliases = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["change-1"],
                    "process_date": ["2026-02-01"],
                    "old_symbol": ["OLD"],
                    "new_symbol": ["NEW"],
                }
            )
        )
        current = pd.DataFrame({"ticker": ["NEW"], "sector": ["Information Technology"]})
        universe, _ = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            symbol_changes=aliases,
            start_date=date(2025, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )
        old = universe[universe["ticker"] == "OLD"].iloc[0]
        new = universe[universe["ticker"] == "NEW"].iloc[0]
        self.assertEqual(pd.Timestamp(old["effective_to_utc"]), pd.Timestamp(new["effective_from_utc"]))

    def test_temporary_when_issued_symbols_are_not_membership_aliases(self) -> None:
        aliases = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["temporary-dotted", "temporary-v-suffix"],
                    "process_date": ["2020-04-01", "2020-11-17"],
                    "old_symbol": ["HWM.WI", "VTRSV"],
                    "new_symbol": ["HWM", "VTRS"],
                    "old_cusip": ["443201108", "92556V106"],
                    "new_cusip": ["443201108", "92556V106"],
                }
            )
        )

        self.assertEqual(aliases, [])

    def test_merger_carries_membership_but_resets_security_identity(self) -> None:
        transitions = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["para-to-psky"],
                    "process_date": ["2025-08-07"],
                    "effective_date": ["2025-08-07"],
                    "old_symbol": ["PARA"],
                    "new_symbol": ["PSKY"],
                    "old_cusip": ["92556H206"],
                    "new_cusip": ["69932A204"],
                    "transition_type": ["stock_merger"],
                    "identity_continuity": [False],
                    "membership_continuity": [True],
                }
            )
        )
        current = pd.DataFrame(
            {
                "ticker": ["PSKY"],
                "company": ["Paramount Skydance"],
                "sector": ["Communication Services"],
            }
        )

        universe, _ = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=[],
            symbol_changes=transitions,
            start_date=date(2025, 1, 1),
            cutoff_date=date(2026, 1, 1),
            anchor_source="anchor.csv",
        )

        self.assertEqual(set(universe["ticker"]), {"PARA", "PSKY"})
        self.assertEqual(universe["security_id"].nunique(), 2)

    def test_announcement_ticker_resolves_through_prior_parallel_symbol_changes(self) -> None:
        html = _announcement_html().replace("January 5, 2026", "March 3, 2020").replace("NEW", "GDI")
        changes = [
            item
            for item in parse_sp500_changes(
                html,
                source_url="https://press.spglobal.com/2020-02-27-gdi",
                published_date=date(2020, 2, 27),
            )
            if item.action == "addition"
        ]
        aliases = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["gdi-to-ir", "ir-to-tt"],
                    "process_date": ["2020-03-02", "2020-03-02"],
                    "old_symbol": ["GDI", "IR"],
                    "new_symbol": ["IR", "TT"],
                }
            )
        )
        current = pd.DataFrame(
            {
                "ticker": ["IR", "TT"],
                "sector": ["Industrials", "Industrials"],
            }
        )

        universe, audit = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            symbol_changes=aliases,
            start_date=date(2019, 7, 1),
            cutoff_date=date(2020, 7, 1),
            anchor_source="anchor.csv",
        )

        new_ir = universe[(universe["ticker"] == "IR") & (universe["effective_from_utc"] > pd.Timestamp("2020-03-02", tz="UTC"))]
        self.assertEqual(len(new_ir), 1)
        self.assertEqual(universe.loc[universe["ticker"].eq("IR"), "security_id"].nunique(), 2)
        self.assertEqual(
            audit["resolved_announcement_tickers"],
            [
                {
                    "effective_at_utc": "2020-03-03T05:00:00+00:00",
                    "source_ticker": "GDI",
                    "effective_ticker": "IR",
                }
            ],
        )

    def test_metadata_propagates_forward_to_resolved_deletion_ticker(self) -> None:
        html = _announcement_html().replace("January 5, 2026", "January 5, 2020").replace("NEW", "OLD")
        changes = parse_sp500_changes(
            html,
            source_url="https://press.spglobal.com/2020-01-01-old-ticker",
            published_date=date(2020, 1, 1),
        )
        deletion = replace(
            next(item for item in changes if item.action == "deletion"),
            effective_at_utc=datetime.fromisoformat("2020-02-05T05:00:00+00:00"),
        )
        addition = next(item for item in changes if item.action == "addition")
        aliases = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["old-to-new"],
                    "process_date": ["2020-01-15"],
                    "old_symbol": ["OLD"],
                    "new_symbol": ["NEW"],
                }
            )
        )
        current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Industrials"]})

        universe, _ = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=[addition, deletion],
            symbol_changes=aliases,
            start_date=date(2019, 7, 1),
            cutoff_date=date(2020, 7, 1),
            anchor_source="anchor.csv",
        )

        self.assertIn("NEW", set(universe["ticker"]))

    def test_old_alias_does_not_rewrite_a_later_reused_ticker(self) -> None:
        html = _announcement_html().replace("January 5, 2026", "May 7, 2026").replace("OLD", "CTRA")
        deletion = [
            item
            for item in parse_sp500_changes(
                html,
                source_url="https://press.spglobal.com/2026-05-01-ctra-deletion",
                published_date=date(2026, 5, 1),
            )
            if item.action == "deletion"
        ]
        aliases = symbol_changes_from_transitions(
            pd.DataFrame(
                {
                    "id": ["old-ctra-to-amr", "cog-to-new-ctra"],
                    "process_date": ["2021-02-04", "2021-10-04"],
                    "old_symbol": ["CTRA", "COG"],
                    "new_symbol": ["AMR", "CTRA"],
                }
            )
        )
        current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Industrials"]})

        universe, audit = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=deletion,
            symbol_changes=aliases,
            start_date=date(2019, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )

        self.assertIn("CTRA", set(universe["ticker"]))
        self.assertEqual(audit["resolved_announcement_tickers"], [])

    def test_current_share_classes_have_distinct_security_identities(self) -> None:
        current = pd.DataFrame(
            {
                "ticker": ["GOOG", "GOOGL"],
                "company": ["Alphabet Class C", "Alphabet Class A"],
                "sector": ["Communication Services", "Communication Services"],
                "CIK": [1652044, 1652044],
            }
        )

        universe, _ = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=[],
            start_date=date(2025, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )

        self.assertEqual(universe["security_id"].nunique(), 2)


def _announcement_html() -> str:
    return """
    <html><body><table>
      <tr><th>Effective Date</th><th>Index Name</th><th>Action</th><th>Company Name</th><th>Ticker</th><th>GICS Sector</th></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P 500</td><td>Addition</td><td>New Company</td><td>NEW</td><td>Information Technology</td></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P 500</td><td>Deletion</td><td>Old Company</td><td>OLD</td><td>Industrials</td></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P MidCap 400</td><td>Addition</td><td>Other</td><td>OTHER</td><td>Industrials</td></tr>
    </table></body></html>
    """


def _legacy_announcement_html() -> str:
    return """
    <html><body><div class="wd_news_body">
      <p>
        T-Mobile US Inc (NASD: TMUS) will replace Red Hat Inc. (NYSE: RHT)
        in the S&amp;P 500 effective prior to the open of trading on Monday, July 15.
      </p>
      <table>
        <tr><td>S&amp;P 500 INDEX – July 15, 2019</td></tr>
        <tr><td></td><td>COMPANY</td><td>GICS ECONOMIC SECTOR</td><td>GICS SUB-INDUSTRY</td></tr>
        <tr><td>ADDED</td><td>T-Mobile US</td><td>Communication Services</td><td>Wireless</td></tr>
        <tr><td>DELETED</td><td>Red Hat</td><td>Information Technology</td><td>Software</td></tr>
      </table>
    </div></body></html>
    """


if __name__ == "__main__":
    unittest.main()
