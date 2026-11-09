"""Observed provider currency corroboration for unchanged archived dividend records."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dividend_currencies import _json
from market_predictor.research.investment_features import FLAGS, ArtifactReference, _object, _publish, _read
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence

MAXIMUM_RESPONSE_BYTES = 2 * 1024**2
BATCH_SIZE = 50
SCHEMA = "market_predictor.investment_dividend_currencies"


def _match_record(original: dict[str, Any], refreshed: dict[str, Any]) -> str | None:
    for field in ("id", "symbol", "ex_date", "rate", "foreign", "special"):
        if field not in original or field not in refreshed or original[field] != refreshed[field]:
            return f"dividend_record_changed_or_missing:{field}"
    if refreshed.get("foreign") is not False or refreshed.get("special") is not False:
        return "dividend_flags_not_explicit_ordinary_domestic"
    for field in ("cancelled", "canceled", "revised", "revision", "due_bill_redemption_date"):
        if refreshed.get(field) not in (None, "", False):
            return f"dividend_refresh_unsupported_terms:{field}"
    for field, value in original.items():
        if field != "currency" and (field not in refreshed or refreshed[field] != value):
            return f"dividend_record_changed_or_missing:{field}"
    if refreshed.get("currency") != "USD":
        return "currency_missing_or_not_USD"
    return None


def _decode_response(
    payload: dict[str, Any], originals: dict[str, dict[str, Any]]
) -> tuple[dict[str, tuple[dict[str, Any], str]], dict[str, str]]:
    if payload.get("next_page_token"):
        raise DataReadinessError("currency ID lookup unexpectedly paginated; no partial page admitted")
    families = payload.get("corporate_actions")
    if not isinstance(families, dict):
        raise DataReadinessError("currency response lacks corporate_actions object")
    seen: set[str] = set()
    resolved: dict[str, tuple[dict[str, Any], str]] = {}
    gaps: dict[str, str] = {}
    count = 0
    for family, records in families.items():
        if not isinstance(records, list):
            raise DataReadinessError("currency response action family is not an array")
        for position, record in enumerate(records):
            count += 1
            if count > BATCH_SIZE or not isinstance(record, dict):
                raise DataReadinessError("currency response exceeds ID batch bound or contains invalid record")
            action_id = record.get("id")
            if not isinstance(action_id, str) or action_id not in originals or action_id in seen:
                raise DataReadinessError("currency response contains unrequested or duplicate action ID")
            seen.add(action_id)
            if family != "cash_dividends":
                gaps[action_id] = "dividend_family_changed"
                continue
            reason = _match_record(originals[action_id], record)
            if reason is None:
                resolved[action_id] = (record, f"corporate_actions.cash_dividends[{position}]")
            else:
                gaps[action_id] = reason
    for action_id in originals.keys() - seen:
        gaps[action_id] = "requested_dividend_not_returned"
    return resolved, gaps


def collect_investment_dividend_currencies(
    *, root: Path, action_config: SourcePin, action_archive: Path, action_audit_sha256: str, output: Path, settings: Settings | None = None,
    required_events: SourcePin | None = None
) -> dict[str, Any]:
    """Refresh IDs only; unchanged rate/entitlement evidence is never replaced.

    The replay owns its own lease and therefore precedes the collection lease.
    Historical dollars are not inferred from ticker, exchange, or issuer domicile.
    """
    root, output = root.resolve(), output.resolve()
    if (output / "final").exists():
        raise FileExistsError("completed immutable dividend currency artifact exists")
    config = inside(root, action_config.path)
    if file_sha256(config) != action_config.sha256:
        raise DataReadinessError("dividend currency action config pin differs")
    evidence = load_corporate_action_evidence(
        root=root,
        config=config,
        archive=action_archive,
        expected_audit_sha256=action_audit_sha256,
        memory_guard=lambda: guard("dividend currency source replay"),
    )
    with heavy_job_lease("collect-investment-dividend-currencies", runtime_dir=root / heavy_job_runtime_dir()):
        guard("dividend currency collection inputs")
        scope: dict[str, str] | None = None
        if required_events is not None:
            scope_payload = _json(_read(root, ArtifactReference.model_validate(required_events.model_dump()), {}))
            if scope_payload.get("schema") != "market_predictor.investment_dividend_event_scope":
                raise DataReadinessError("explicit dividend event scope schema differs")
            items = scope_payload.get("events")
            if not isinstance(items, list) or not 1 <= len(items) <= 10_000:
                raise DataReadinessError("explicit dividend event scope exceeds bounded population")
            scope = {}
            for item in items:
                if not isinstance(item, dict) or set(item) != {"action_id", "original_record_sha256"}:
                    raise DataReadinessError("explicit dividend event scope record differs")
                if not isinstance(item["action_id"], str) or item["action_id"] in scope:
                    raise DataReadinessError("explicit dividend event scope repeats action IDs")
                scope[item["action_id"]] = item["original_record_sha256"]
        originals: dict[str, dict[str, Any]] = {}
        for families in evidence.records_by_symbol.values():
            for record in families.get("cash_dividends", []):
                if (
                    record.get("currency") not in {None, ""}
                    or record.get("foreign") is not False
                    or record.get("special") is not False
                    or (scope is None and not "2019-07-09" <= str(record.get("ex_date", "")) <= "2024-05-28")
                    or (scope is not None and record.get("id") not in scope)
                ):
                    continue
                action_id = record.get("id")
                if not isinstance(action_id, str) or not action_id:
                    raise DataReadinessError("archived dividend lacks stable action ID")
                if action_id in originals and originals[action_id] != record:
                    raise DataReadinessError("archived dividend ID has conflicting records")
                if scope is not None and scope[action_id] != json_sha256(record):
                    raise DataReadinessError("explicit dividend event scope original hash differs")
                originals[action_id] = record
        if scope is not None and set(originals) != set(scope):
            raise DataReadinessError("explicit dividend event scope contains unsupported or missing actions")
        pins = {str(inside(root, name)): digest for name, digest in evidence.source_files.items()}
        pins[str(config)] = action_config.sha256
        if required_events is not None:
            pins[str(inside(root, required_events.path))] = required_events.sha256
        pins[str(Path(__file__).resolve())] = file_sha256(Path(__file__))
        request = {
            "action_config": action_config.model_dump(),
            "action_archive": str(action_archive),
            "action_audit_sha256": action_audit_sha256,
            "original_records_sha256": json_sha256(originals),
            "implementation_sha256": pins[str(Path(__file__).resolve())],
            "batch_size": BATCH_SIZE,
            "purpose": "observed_currency_verification_only_no_historical_clock_backdating",
            **({"required_events": required_events.model_dump()} if required_events is not None else {}),
        }
        job_sha = json_sha256(request)
        output.mkdir(parents=True, exist_ok=True)
        job_path = output / "_job.json"
        if job_path.exists():
            if json.loads(job_path.read_text()) != request:
                raise DataReadinessError("dividend currency resume request differs")
        else:
            write_json_object(job_path, request)
        pins[str(job_path)] = file_sha256(job_path)
        source = AlpacaSource(settings or Settings())
        resolved_rows: list[dict[str, Any]] = []
        unresolved: list[dict[str, str]] = []
        ids = sorted(originals)
        for start in range(0, len(ids), BATCH_SIZE):
            guard("dividend currency ID batch")
            batch_ids = ids[start : start + BATCH_SIZE]
            batch = {key: originals[key] for key in batch_ids}
            directory = output / "batches" / json_sha256(batch_ids)[:24]
            published_checkpoint = directory.exists()
            try:
                if not directory.exists():
                    response = source.client.get_bytes_with_metadata(
                        source.corporate_actions_url,
                        headers=source.headers,
                        params={"ids": ",".join(batch_ids)},
                        maximum_body_bytes=MAXIMUM_RESPONSE_BYTES,
                    )
                    if response.status_code != 200 or len(response.body) > MAXIMUM_RESPONSE_BYTES:
                        raise DataReadinessError("dividend currency response status or byte bound differs")
                    metadata = asdict(response)
                    metadata.pop("body")
                    metadata["retrieved_at_utc"] = response.retrieved_at_utc.isoformat()

                    def save(path: Path, body: bytes = response.body, transport: dict[str, Any] = metadata) -> None:
                        (path / "response.bin").write_bytes(body)
                        write_json_object(path / "transport.json", transport)

                    _publish(
                        directory,
                        {
                            "schema": SCHEMA + ".batch",
                            "status": "currency_response_archived",
                            "job_sha256": job_sha,
                            "requested_ids": batch_ids,
                            **FLAGS,
                        },
                        {},
                        save,
                    )
                manifest_ref = ArtifactReference(path=str(directory / "_manifest.json"), sha256=file_sha256(directory / "_manifest.json"))
                receipt = _object(root, manifest_ref, pins)
                if receipt.get("job_sha256") != job_sha or receipt.get("requested_ids") != batch_ids:
                    raise DataReadinessError("dividend currency checkpoint identity differs")
                if set(receipt.get("files", {})) != {"response.bin", "transport.json"}:
                    raise DataReadinessError("dividend currency checkpoint files differ")
                body_ref = ArtifactReference(path=str(directory / "response.bin"), sha256=receipt["files"]["response.bin"])
                body = _read(root, body_ref, pins)
                transport = _object(
                    root, ArtifactReference(path=str(directory / "transport.json"), sha256=receipt["files"]["transport.json"]), pins
                )
                if (
                    transport.get("sha256") != body_ref.sha256
                    or transport.get("body_length") != len(body)
                    or transport.get("status_code") != 200
                ):
                    raise DataReadinessError("dividend currency transport differs from response bytes")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise DataReadinessError("dividend currency response is not object")
                resolved, gaps = _decode_response(payload, batch)
                for action_id, (_record, locator) in sorted(resolved.items()):
                    resolved_rows.append(
                        {
                            "action_id": action_id,
                            "original_record_sha256": json_sha256(originals[action_id]),
                            "response": body_ref.model_dump(),
                            "transport": {"path": str(directory / "transport.json"), "sha256": receipt["files"]["transport.json"]},
                            "record_locator": locator,
                            "retrieved_at_utc": transport["retrieved_at_utc"],
                        }
                    )
                unresolved.extend({"action_id": action_id, "reason": reason} for action_id, reason in sorted(gaps.items()))
            except MemoryBudgetError:
                raise
            except (OSError, ValueError, RuntimeError, DataReadinessError) as exc:
                # Corrupted immutable receipt/body is a refusal, not a provider gap.
                if published_checkpoint:
                    raise
                unresolved.extend({"action_id": action_id, "reason": str(exc)} for action_id in batch_ids)
        evidence.recheck(root)
        report = {
            "schema": SCHEMA,
            "status": "currency_evidence_complete_with_gaps" if unresolved else "currency_evidence_complete",
            "job_sha256": job_sha,
            "request": request,
            "source_files": pins,
            "requested_actions": len(originals),
            "resolved_USD_actions": len(resolved_rows),
            "unresolved_actions": len(unresolved),
            "availability_policy": "actual_response_retrieval_only",
            **FLAGS,
        }

        def final_save(path: Path) -> None:
            write_json_object(path / "originals.json", {"records": originals})
            write_json_object(path / "resolved.json", {"records": resolved_rows, "unresolved": unresolved})

        return _publish(output / "final", report, pins, final_save)


def load_investment_currency_evidence(
    root: Path, manifest: Path, sha256: str, pins: dict[str, str], *, interpretation_policy_sha256: str
) -> dict[str, tuple[dict[str, Any], EvidenceReference]]:
    """Independently replay raw responses and exact archived originals before admission."""
    report = _object(root, ArtifactReference(path=str(manifest), sha256=sha256), pins)
    if report.get("schema") != SCHEMA or report.get("status") not in {"currency_evidence_complete", "currency_evidence_complete_with_gaps"}:
        raise DataReadinessError("not a completed dividend currency artifact")
    for name, digest in report["source_files"].items():
        path = Path(name) if Path(name).is_absolute() else inside(root, name)
        if file_sha256(path) != digest:
            raise DataReadinessError("dividend currency source changed")
        pins[str(path)] = digest
    parent = Path(manifest).parent
    data = {
        name: _object(root, ArtifactReference(path=str(parent / name), sha256=digest), pins) for name, digest in report["files"].items()
    }
    originals = data["originals.json"]["records"]
    if json_sha256(originals) != report["request"]["original_records_sha256"]:
        raise DataReadinessError("dividend currency original record inventory differs")
    rows = data["resolved.json"]["records"]
    gaps = data["resolved.json"]["unresolved"]
    resolved_ids = [row["action_id"] for row in rows]
    gap_ids = [row["action_id"] for row in gaps]
    if (
        len(originals) != report["requested_actions"]
        or len(rows) != report["resolved_USD_actions"]
        or len(gaps) != report["unresolved_actions"]
        or len(set(resolved_ids)) != len(rows)
        or len(set(gap_ids)) != len(gaps)
        or set(resolved_ids).intersection(gap_ids)
        or set(resolved_ids).union(gap_ids) != set(originals)
    ):
        raise DataReadinessError("dividend currency population accounting differs")
    result = {}
    cache: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
    for row in data["resolved.json"]["records"]:
        action_id = row["action_id"]
        reference = ArtifactReference.model_validate_json(json.dumps(row["response"]))
        transport_ref = ArtifactReference.model_validate_json(json.dumps(row["transport"]))
        if (
            report["source_files"].get(reference.path) != reference.sha256
            or report["source_files"].get(transport_ref.path) != transport_ref.sha256
        ):
            raise DataReadinessError("dividend currency response or transport is outside source inventory")
        transport = _object(root, transport_ref, pins)
        if (
            transport.get("retrieved_at_utc") != row["retrieved_at_utc"]
            or transport.get("sha256") != reference.sha256
            or transport.get("status_code") != 200
        ):
            raise DataReadinessError("dividend currency retrieval clock/transport binding differs")
        if reference.path not in cache:
            payload = _object(root, reference, pins)
            # The response may contain other requested IDs from the same batch.
            cache[reference.path] = _decode_response(payload, originals)[0]
        record, locator = cache[reference.path][action_id]
        if action_id in result or json_sha256(originals[action_id]) != row["original_record_sha256"] or locator != row["record_locator"]:
            raise DataReadinessError("dividend currency record binding differs")
        retrieved = datetime.fromisoformat(row["retrieved_at_utc"])
        derived = {**record, "original_record_sha256": row["original_record_sha256"]}
        result[action_id] = (
            derived,
            EvidenceReference(
                reference=inside(root, reference.path).relative_to(root.resolve()).as_posix(),
                artifact_sha256=reference.sha256,
                record_locator=locator,
                interpretation_policy_sha256=interpretation_policy_sha256,
                retrieved_at=retrieved,
                available_at=retrieved,
            ),
        )
    return result
