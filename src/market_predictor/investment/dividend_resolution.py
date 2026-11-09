"""Exact required-event dividend resolution using immutable captured authorities."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dividend_currencies import (
    CapturedCurrencySource,
    CurrencyCorroboration,
    CurrencyCorroborationManifest,
    DividendAmountCorrectionManifest,
    DividendEvidenceUnavailable,
    IssuerDividendAmountCorrection,
    OfficialDocumentIdentity,
    StockCompositeIdentity,
    _decimal,
    _isin,
    _json,
    _workbook_rows,
)
from market_predictor.research.investment_features import FLAGS, _publish
from market_predictor.sources.ictax import verify_ictax_archive
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

MAXIMUM_MANIFEST_BYTES = 64 * 1024**2


class RequiredDividendEvent(HoldingContract):
    action_id: str
    symbol: str
    ex_date: date
    original_record_sha256: str
    source: SourcePin


class DividendSecurityIdentity(HoldingContract):
    symbol: str
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{9}$")]
    isin: Annotated[str, Field(pattern=r"^US[A-Z0-9]{9}[0-9]$")]
    official_document: OfficialDocumentIdentity | None = None
    issuer_workbook: CapturedCurrencySource | None = None
    # The stock identity contract is resolved by the same closed discriminator
    # as CurrencyCorroboration, never an operator boolean approval.
    stock_composite: StockCompositeIdentity | None = None

    @model_validator(mode="after")
    def exactly_one_authority(self) -> DividendSecurityIdentity:
        if sum(x is not None for x in (self.official_document, self.issuer_workbook, self.stock_composite)) != 1:
            raise ValueError("dividend security requires exactly one identity authority")
        if _isin(self.cusip) != self.isin:
            raise ValueError("dividend identity CUSIP/ISIN checksum differs")
        return self


class DividendResolutionRequest(HoldingContract):
    schema_name: Literal["market_predictor.investment_dividend_resolution_request"] = Field(alias="schema")
    events: Annotated[tuple[RequiredDividendEvent, ...], Field(min_length=1, max_length=10_000)]
    identities: Annotated[tuple[DividendSecurityIdentity, ...], Field(min_length=1, max_length=1024)]
    tax_archives: Annotated[tuple[SourcePin, ...], Field(min_length=1, max_length=64)]
    preserved_corroborations: SourcePin
    preserved_amount_corrections: SourcePin
    approved_new_amount_corrections: SourcePin | None = None

    @model_validator(mode="after")
    def unique_population(self) -> DividendResolutionRequest:
        if len({e.action_id for e in self.events}) != len(self.events):
            raise ValueError("required dividend action IDs repeat")
        keys = [(i.symbol, i.cusip) for i in self.identities]
        if len(set(keys)) != len(keys):
            raise ValueError("dividend security identities repeat")
        return self


def _semantic(value: Any) -> Any:
    """Preserve all PAYMENT facts; normalize numeric scale without rounding."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, Decimal)):
        return {"number": format(Decimal(value).normalize(), "f")}
    if isinstance(value, list):
        return [_semantic(x) for x in value]
    if isinstance(value, dict):
        return {k: _semantic(v) for k, v in value.items()}
    raise DataReadinessError("unsupported tax PAYMENT scalar")


def resolve_investment_dividends(*, root: Path, request: SourcePin, output: Path) -> dict[str, Any]:
    """Resolve each unique requested action; unsupported facts remain local gaps.

    Verify archives under their own lease first, then acquire the resolution lease.
    Existing assertions and corrections retain their exact JSON objects and source
    bytes. New assertions always replay the complete typed evidence adapter.
    """
    root, output = root.resolve(), inside(root.resolve(), output)
    pins: dict[str, str] = {}
    producer = Path(__file__).resolve()
    producer_paths = (producer, producer.with_name("dividend_currencies.py"), producer.with_name("stock_identity.py"),
        producer.parents[1] / "universe/security_class_evidence.py", producer.parents[1] / "sources/ictax.py")
    implementations = {str(path): file_sha256(path) for path in producer_paths}

    def read(pin: SourcePin, maximum: int = MAXIMUM_MANIFEST_BYTES) -> bytes:
        guard("dividend resolution source read")
        path = inside(root, pin.path)
        if path.stat().st_size > maximum:
            raise DataReadinessError("dividend resolution source exceeds byte bound")
        raw = path.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != pin.sha256:
            raise DataReadinessError("dividend resolution source hash differs")
        name = str(path)
        if name in pins and pins[name] != pin.sha256:
            raise DataReadinessError("dividend resolution source has conflicting hashes")
        pins[name] = pin.sha256
        return raw

    with heavy_job_lease("prepare-investment-dividend-resolution", runtime_dir=root / heavy_job_runtime_dir()):
        request_raw = read(request)
        _json(request_raw)
        policy = DividendResolutionRequest.model_validate_json(request_raw)
    archives = []
    for archive_pin in policy.tax_archives:
        archives.append(verify_ictax_archive(root=root, manifest=inside(root, archive_pin.path), manifest_sha256=archive_pin.sha256))
    with heavy_job_lease("resolve-investment-dividend-events", runtime_dir=root / heavy_job_runtime_dir()):
        guard("dividend event resolution preparation")
        if output.exists():
            raise FileExistsError("immutable dividend resolution output already exists")
        read(request)
        seed_raw = read(policy.preserved_corroborations)
        seed = json.loads(seed_raw)
        CurrencyCorroborationManifest.model_validate_json(seed_raw)
        merged = {"schema": seed["schema"], "corroborations": list(seed["corroborations"])}
        supported = {x["action_id"]: x for x in merged["corroborations"]}
        if len(supported) != len(merged["corroborations"]):
            raise DataReadinessError("preserved currency assertions repeat action IDs")
        correction_raw = read(policy.preserved_amount_corrections)
        corrections = DividendAmountCorrectionManifest.model_validate_json(correction_raw)
        approved = {c.action_id: c for c in corrections.corrections}
        correction_json = json.loads(correction_raw)
        if policy.approved_new_amount_corrections is not None:
            extra_raw = read(policy.approved_new_amount_corrections)
            extra = DividendAmountCorrectionManifest.model_validate_json(extra_raw)
            frozen = {e.action_id for e in policy.events}
            for model, obj in zip(extra.corrections, json.loads(extra_raw)["corrections"], strict=True):
                if model.action_id not in frozen or model.action_id in approved:
                    raise DataReadinessError("new amount correction is duplicate or outside required events")
                approved[model.action_id] = model
                correction_json["corrections"].append(obj)
        identities = {(i.symbol, i.cusip): i for i in policy.identities}
        index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for manifest, archive in zip(policy.tax_archives, archives, strict=True):
            read(manifest)
            for name, digest in archive["source_files"].items():
                if name in pins and pins[name] != digest:
                    raise DataReadinessError("tax archive source pin conflicts")
                pins[name] = digest
            for unit in archive["records"]:
                guard("dividend tax PAYMENT index")
                if unit["status"] not in ("responses_archived", "retained_sources_verified"):
                    continue
                tax_source = unit["sources"]["structured"]["body"]
                payload = _json(read(SourcePin.model_validate(tax_source), 2 * 1024**2))
                entries = payload.get("data", {}).get("calculationDetails", {}).get("entries", [])
                if not isinstance(entries, list) or len(entries) > 128:
                    continue
                for number, event in enumerate(entries):
                    if not isinstance(event, dict) or event.get("type") != "PAYMENT":
                        continue
                    stamp = event.get("exDate")
                    if type(stamp) is not int or stamp % 86_400_000:
                        continue
                    day = datetime.fromtimestamp(stamp / 1000, UTC).date().isoformat()
                    index[unit["isin"], day].append({"event": event, "entry_index": number,
                        "sources": unit["sources"], "year": unit["year"], "signature": json_sha256(_semantic(event)),
                        "lookup": unit.get("lookup"), "same_exdate_count": sum(isinstance(e, dict) and e.get("type") == "PAYMENT"
                            and e.get("exDate") == stamp for e in entries)})
        workbook_cache: dict[str, dict[int, dict[str, str]]] = {}
        page_cache: dict[str, dict[str, Any]] = {}
        results, pending = [], []
        stock_identity_cache: dict[str, Any] = {}
        for required in policy.events:
            guard("required dividend source replay")
            row: dict[str, Any] = {"action_id": required.action_id, "symbol": required.symbol,
                "ex_date": required.ex_date.isoformat(), "status": "unresolved", "reasons": []}
            results.append(row)
            key = required.source.sha256
            if key not in page_cache:
                if len(page_cache) >= 32:
                    page_cache.clear()
                page_cache[key] = parse_strict_json_object(read(required.source, 8 * 1024**2), label="original dividend page")
            records = page_cache[key].get("corporate_actions", {}).get("cash_dividends", [])
            matches = [r for r in records if r.get("id") == required.action_id]
            if len(matches) != 1 or json_sha256(matches[0]) != required.original_record_sha256:
                raise DataReadinessError("required dividend original record is absent or differs")
            original = matches[0]
            if original.get("symbol") != required.symbol or original.get("ex_date") != required.ex_date.isoformat():
                raise DataReadinessError("required dividend symbol/ex-date differs from original")
            if required.action_id in supported:
                if supported[required.action_id]["original_record_sha256"] != required.original_record_sha256:
                    raise DataReadinessError("preserved assertion original record differs")
                row["status"] = "preserved_assertion"
                continue
            if original.get("foreign") is not False or original.get("special") is not False:
                row["reasons"] = ["ordinary_domestic_cash_entitlement_unproven"]
                continue
            identity = identities.get((required.symbol, original.get("cusip")))
            if identity is None:
                row["reasons"] = ["security_identity_unproven"]
                continue
            candidates = index.get((identity.isin, required.ex_date.isoformat()), [])
            groups = {x["signature"] for x in candidates}
            if len(groups) != 1:
                row["reasons"] = ["tax_payment_absent" if not groups else "tax_payment_versions_conflict"]
                continue
            unique_payload_candidates = [c for c in candidates if c["same_exdate_count"] == 1]
            if not unique_payload_candidates:
                row["reasons"] = ["tax_payment_ex_date_ambiguous"]
                continue
            selected = min(unique_payload_candidates, key=lambda c: (c["sources"]["structured"]["body"]["sha256"], c["entry_index"]))
            event = selected["event"]
            if (event.get("currency", {}).get("shortName") != "USD"
                or type(event.get("nominalQuantity")) is not int or event["nominalQuantity"] != 1):
                row["reasons"] = ["tax_payment_usd_per_share_basis_unproven"]
                continue
            # Capture corruption is fatal, including when semantic admission later
            # finds a local gap. Validate all paired source bytes eagerly.
            for source_pair in selected["sources"].values():
                for source_pin in source_pair.values():
                    read(SourcePin.model_validate(source_pin))
            parsed: CurrencyCorroboration | None = None
            try:
                amount, tax_amount = _decimal(original.get("rate")), _decimal(event.get("value"))
                issuer = None
                if identity.issuer_workbook is not None:
                    source = identity.issuer_workbook
                    wh = source.body.sha256
                    if wh not in workbook_cache:
                        workbook_cache[wh] = _workbook_rows(read(SourcePin.model_validate(source.body.model_dump()), 16 * 1024**2))
                    found = [(n, c) for n, c in workbook_cache[wh].items()
                        if c.get("B") == required.symbol and c.get("C") == identity.cusip
                        and c.get("D") == required.ex_date.strftime("%m/%d/%Y")]
                    if len(found) != 1:
                        raise DividendEvidenceUnavailable("issuer distribution row missing or ambiguous")
                    number, cells = found[0]
                    issuer = {"source": source.model_dump(mode="json"), "row": number}
                    authority: dict[str, Any] = {"kind": "issuer_workbook", **issuer, "cusip": identity.cusip}
                    if any(cells.get(c) and _decimal(cells[c]) != 0 for c in ("H", "I")):
                        raise DividendEvidenceUnavailable("issuer distribution contains capital gains")
                    gross = _decimal(cells.get("G"))
                    if gross != amount and required.action_id not in approved:
                        correction = {"method": "issuer_gross_amount_correction", "action_id": required.action_id,
                            "original_record_sha256": required.original_record_sha256, "symbol": required.symbol,
                            "ex_date": required.ex_date.isoformat(), "cusip": identity.cusip, "isin": identity.isin,
                            "original_rate": str(amount), "corrected_gross_amount": str(gross), "issuer_rate": issuer}
                        parsed_correction = IssuerDividendAmountCorrection.model_validate_json(json.dumps(correction))
                        parsed_correction.resolve_against(original, root, request.sha256)
                        pending.append(correction)
                        row["reasons"] = ["issuer_provider_amount_conflict_pending_review"]
                        continue
                elif identity.official_document is not None:
                    authority = identity.official_document.model_dump(mode="json")
                else:
                    authority = identity.stock_composite.model_dump(mode="json") if identity.stock_composite is not None else {}
                correction_model = approved.get(required.action_id)
                effective = (amount if correction_model is None else
                    correction_model.resolve_against(original, root, request.sha256).gross_amount)
                method = "exact_tax_event" if correction_model is None and amount == tax_amount else "issuer_rate_tax_currency_bridge"
                if method != "exact_tax_event" and issuer is None:
                    raise DividendEvidenceUnavailable("exact original amount differs from tax gross distribution")
                assertion: dict[str, Any] = {"method": method, "action_id": required.action_id, "symbol": required.symbol,
                    "original_record_sha256": required.original_record_sha256, "ex_date": required.ex_date.isoformat(),
                    "currency": "USD", "identity": authority, "issuer_rate": issuer if method != "exact_tax_event" else None,
                    "tax_value_difference": str(tax_amount - amount), "tax": {"source": selected["sources"]["structured"],
                        "gross_context": selected["sources"]["gross_context"], "isin": identity.isin,
                        "entry_index": selected["entry_index"], "gross_amount": str(tax_amount)}}
                if correction_model is not None:
                    assertion["tax_resolved_value_difference"] = str(tax_amount - effective)
                if identity.stock_composite is not None:
                    lookup_pin = selected.get("lookup")
                    if lookup_pin is None:
                        raise DividendEvidenceUnavailable("stock tax classification source missing")
                    lookup_manifest = json.loads(read(SourcePin.model_validate(lookup_pin)))
                    lookup_directory = inside(root, lookup_pin["path"]).parent
                    assertion["tax"]["lookup"] = {
                        "body": {"path": str(lookup_directory / "response.bin"),
                            "sha256": lookup_manifest["files"]["response.bin"]},
                        "receipt": {"path": str(lookup_directory / "receipt.json"),
                            "sha256": lookup_manifest["files"]["receipt.json"]}}
                parsed = CurrencyCorroboration.model_validate_json(json.dumps(assertion))
                refs = parsed.validates_against(original, root, request.sha256,
                    amount_correction=correction_model, stock_identity_cache=stock_identity_cache)
                for ref in refs:
                    pins[str(inside(root, ref.reference))] = ref.artifact_sha256
                supported[required.action_id] = assertion
                merged["corroborations"].append(assertion)
                row.update(status="new_assertion_source_validated", method=method)
                if isinstance(parsed.identity, StockCompositeIdentity):
                    row["stock_tax_identity_observation"] = parsed.identity.tax_identity_observation
            except DividendEvidenceUnavailable as exc:
                row["reasons"] = [str(exc)]
                if parsed is not None and isinstance(parsed.identity, StockCompositeIdentity):
                    row["stock_tax_identity_observation"] = parsed.identity.tax_identity_observation
        if merged["corroborations"][:len(seed["corroborations"])] != seed["corroborations"]:
            raise DataReadinessError("preserved dividend assertions changed")
        CurrencyCorroborationManifest.model_validate_json(json.dumps(merged))
        for name, digest in stock_identity_cache.get("source_files", {}).items():
            if name in pins and pins[name] != digest:
                raise DataReadinessError("stock dividend identity source hash conflicts")
            pins[name] = digest
        for name, digest in pins.items():
            guard("dividend publication source recheck")
            if file_sha256(inside(root, name)) != digest:
                raise DataReadinessError("dividend source changed before publication")
        for name, digest in implementations.items():
            if file_sha256(Path(name)) != digest:
                raise DataReadinessError("dividend producer changed before publication")
        report = {"implementation_files": implementations, "schema": "market_predictor.investment_dividend_resolution",
            "status": "complete_with_unavailable"
            if any(r["status"] == "unresolved" for r in results) else "complete", "request": request.model_dump(),
            "required_events": len(results), "new_supported_events": sum(r["status"] == "new_assertion_source_validated" for r in results),
            "unresolved_events": sum(r["status"] == "unresolved" for r in results),
            "preserved_assertions": len(seed["corroborations"]), "total_supported_assertions": len(merged["corroborations"]),
            "pending_amount_corrections": len(pending), "source_files": pins, **FLAGS}
        def save(path: Path) -> None:
            write_json_object(path / "currency-corroborations.json", merged)
            if policy.approved_new_amount_corrections is None:
                (path / "amount-corrections.json").write_bytes(correction_raw)
            else:
                write_json_object(path / "amount-corrections.json", correction_json)
            write_json_object(path / "event-coverage.json", {"events": results})
            write_json_object(path / "pending-amount-corrections.json", {"corrections": pending})
        return _publish(output, report, pins, save)
