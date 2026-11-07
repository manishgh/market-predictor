"""SEC CompanyFacts decoding only; publication clocks must come from accession evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from market_predictor.investment.fundamentals import Concept, FilingFact, FilingFactRejection
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256, Timestamp

# Alternatives represent the same economic total: never sum aliases or debt parts.
_CLOCK = TypeAdapter(Timestamp)

TAG_MAPPING: dict[Concept, tuple[str, ...]] = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "net_income": ("NetIncomeLoss",),
    "assets": ("Assets",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "debt": (),
}
# SEC has no universally populated consolidated total debt tag. Do not reinterpret
# long-term debt, current debt or their aliases as total debt.


class SecFactsResult(HoldingContract):
    facts: tuple[FilingFact, ...]
    skipped: tuple[str, ...]
    rejections: tuple[FilingFactRejection, ...] = ()
    research_only: Literal[True] = True


def adapt_sec_company_facts(
    payload: Mapping[str, Any],
    *,
    security_id: str,
    cik: str,
    source_sha256: str,
    retrieved_at: datetime,
    accepted_at_by_accession: Mapping[str, datetime],
    first_observed_at: datetime | None = None,
    source_policy: Literal["observed", "reconstructed_research"] = "reconstructed_research",
) -> SecFactsResult:
    """Accept decoded immutable source plus separately pinned exact accepted clocks.

    SEC `filed` is a date, not an intraday publication timestamp. Unknown accepted
    clocks are reported and omitted. CompanyFacts is issuer-wide consolidated GAAP;
    custom taxonomy and foreign/non-USD units are omitted, never converted.
    """
    TypeAdapter(Sha256).validate_python(source_sha256)
    _CLOCK.validate_python(retrieved_at)
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieval clock must be timezone aware")
    if str(payload.get("cik", "")).lstrip("0") != cik.lstrip("0"):
        raise ValueError("SEC payload CIK differs from security binding")
    if source_policy == "observed" and first_observed_at is None:
        first_observed_at = retrieved_at
    envelope = payload.get("facts")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("us-gaap", {}), dict):
        raise ValueError("SEC fact namespace envelope is malformed")
    namespace = envelope.get("us-gaap", {})
    rejections: list[FilingFactRejection] = []
    skipped: list[str] = ["debt:total_interest_bearing_debt_not_available_from_unambiguous_standard_tag"]
    groups: dict[tuple[str, str, date | None, date], list[FilingFact]] = {}

    def rejection(
        concept: Concept,
        tag: str,
        locator: str,
        row: Mapping[str, Any],
        accepted: datetime,
        detail: str,
        *,
        aliases: tuple[FilingFact, ...] = (),
    ) -> FilingFactRejection:
        start, end = None, None
        parse_error = False
        try:
            end = date.fromisoformat(row["end"])
            if "start" in row:
                start = date.fromisoformat(row["start"])
        except (KeyError, TypeError, ValueError):
            parse_error = True
        instant = concept in {"assets", "cash", "debt"}
        malformed = parse_error or end is None or instant != (start is None) or (start is not None and end < start)
        return FilingFactRejection(
            security_id=security_id,
            cik=cik,
            accession=row["accn"],
            form=row["form"],
            source_sha256=source_sha256,
            source_locator=locator,
            related_source_locators=tuple(sorted(f.source_locator for f in aliases)),
            concept=concept,
            tag=tag,
            period_scope="concept" if malformed else "exact",
            start=start,
            end=end,
            reason="ambiguous_alias_values" if aliases else "invalid_fact",
            detail=detail,
            published_at=accepted,
            retrieved_at=retrieved_at,
            first_observed_at=first_observed_at,
            availability_basis=source_policy,
        )

    for concept, tags in TAG_MAPPING.items():
        for tag in tags:
            entry = namespace.get(tag, {})
            if not isinstance(entry, dict) or not isinstance(entry.get("units", {}), dict):
                raise ValueError("SEC fact tag/unit envelope is malformed")
            units = entry.get("units", {})
            if not isinstance(units.get("USD", []), list):
                raise ValueError("SEC USD fact rows must be an array")
            for unit in units:
                if unit != "USD":
                    skipped.append(f"{tag}:unsupported_unit:{unit}")
            for index, row in enumerate(units.get("USD", [])):
                if not isinstance(row, dict):
                    raise ValueError("SEC fact row envelope is malformed")
                accession = row.get("accn", "")
                accepted = accepted_at_by_accession.get(accession)
                if accepted is None:
                    skipped.append(f"{tag}:{accession}:accepted_clock_unknown")
                    continue
                _CLOCK.validate_python(accepted)
                if row.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                    skipped.append(f"{tag}:{accession}:unsupported_form")
                    continue
                locator = f"facts.us-gaap.{tag}.units.USD[{index}]"
                try:
                    fact = FilingFact(
                        security_id=security_id,
                        cik=cik,
                        accession=accession,
                        form=row["form"],
                        source_sha256=source_sha256,
                        source_locator=locator,
                        concept=concept,
                        tag=tag,
                        period_kind="duration" if "start" in row else "instant",
                        start=date.fromisoformat(row["start"]) if "start" in row else None,
                        end=date.fromisoformat(row["end"]),
                        value=float(row["val"]),
                        published_at=accepted,
                        retrieved_at=retrieved_at,
                        first_observed_at=first_observed_at,
                        availability_basis=source_policy,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    detail = ";".join(str(e["msg"]) for e in exc.errors()) if isinstance(exc, ValidationError) else str(exc)
                    rejections.append(rejection(concept, tag, locator, row, accepted, detail or type(exc).__name__))
                    continue
                key = (concept, accession, fact.start, fact.end)
                groups.setdefault(key, []).append(fact)
    facts: list[FilingFact] = []
    for group in groups.values():
        if len({fact.value for fact in group}) > 1:
            first = group[0]
            rejections.append(
                rejection(
                    first.concept,
                    first.tag,
                    first.source_locator,
                    {
                        "accn": first.accession,
                        "form": first.form,
                        "end": first.end.isoformat(),
                        **({"start": first.start.isoformat()} if first.start else {}),
                    },
                    first.published_at,
                    "conflicting standard tag values for the same filing period",
                    aliases=tuple(group),
                )
            )
        else:
            # Preserve existing declared tag precedence and exact valid source identity.
            facts.append(group[0])
    return SecFactsResult(
        facts=tuple(facts), skipped=tuple(sorted(set(skipped))), rejections=tuple(sorted(rejections, key=lambda r: r.sha256()))
    )
