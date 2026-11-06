"""SEC CompanyFacts decoding only; publication clocks must come from accession evidence."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Literal

from market_predictor.investment.fundamentals import Concept, FilingFact
from market_predictor.swing.contracts.holding_accounting import HoldingContract

# Alternatives represent the same economic total: never sum aliases or debt parts.
TAG_MAPPING: dict[Concept, tuple[str, ...]] = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "net_income": ("NetIncomeLoss",), "assets": ("Assets",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "debt": (),
}
# SEC has no universally populated consolidated total debt tag. Do not reinterpret
# long-term debt, current debt or their aliases as total debt.


class SecFactsResult(HoldingContract):
    facts: tuple[FilingFact, ...]
    skipped: tuple[str, ...]
    research_only: Literal[True] = True


def adapt_sec_company_facts(
    payload: Mapping[str, Any], *, security_id: str, cik: str, source_sha256: str,
    retrieved_at: datetime, accepted_at_by_accession: Mapping[str, datetime],
    first_observed_at: datetime | None = None,
    source_policy: Literal["observed", "reconstructed_research"] = "reconstructed_research",
) -> SecFactsResult:
    """Accept decoded immutable source plus separately pinned exact accepted clocks.

    SEC `filed` is a date, not an intraday publication timestamp. Unknown accepted
    clocks are reported and omitted. CompanyFacts is issuer-wide consolidated GAAP;
    custom taxonomy and foreign/non-USD units are omitted, never converted.
    """
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieval clock must be timezone aware")
    if str(payload.get("cik", "")).lstrip("0") != cik.lstrip("0"):
        raise ValueError("SEC payload CIK differs from security binding")
    if source_policy == "observed" and first_observed_at is None:
        first_observed_at = retrieved_at
    namespace = payload.get("facts", {}).get("us-gaap", {})
    facts: list[FilingFact] = []
    skipped: list[str] = ["debt:total_interest_bearing_debt_not_available_from_unambiguous_standard_tag"]
    seen: dict[tuple[str, str, str | None, str], FilingFact] = {}
    for concept, tags in TAG_MAPPING.items():
        for tag in tags:
            units = namespace.get(tag, {}).get("units", {})
            for unit in units:
                if unit != "USD":
                    skipped.append(f"{tag}:unsupported_unit:{unit}")
            for index, row in enumerate(units.get("USD", [])):
                accession = row.get("accn", "")
                accepted = accepted_at_by_accession.get(accession)
                if accepted is None:
                    skipped.append(f"{tag}:{accession}:accepted_clock_unknown")
                    continue
                if row.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                    skipped.append(f"{tag}:{accession}:unsupported_form")
                    continue
                fact = FilingFact(security_id=security_id, cik=cik, accession=accession, form=row["form"],
                    source_sha256=source_sha256, source_locator=f"facts.us-gaap.{tag}.units.USD[{index}]",
                    concept=concept, tag=tag, period_kind="duration" if "start" in row else "instant",
                    start=date.fromisoformat(row["start"]) if "start" in row else None,
                    end=date.fromisoformat(row["end"]), value=float(row["val"]), published_at=accepted,
                    retrieved_at=retrieved_at, first_observed_at=first_observed_at, availability_basis=source_policy)
                key = (concept, accession, row.get("start"), row["end"])
                previous = seen.get(key)
                if previous is not None:
                    if previous.value != fact.value:
                        raise ValueError("ambiguous standard tag aliases for a filing period")
                    continue
                seen[key] = fact
                facts.append(fact)
    return SecFactsResult(facts=tuple(facts), skipped=tuple(sorted(set(skipped))))
