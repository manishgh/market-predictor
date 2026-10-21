"""One source verifier for reviewed retrospective corporate-action facts."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_materialization import SourcePin


def verify_reviewed_action_sources(root: Path, pin: SourcePin, sources: Mapping[str, str],
    receipts: Iterable[SourcePin], *, memory_guard: Callable[[], None],
) -> dict[str, str]:
    if pin.path in sources:
        raise DataReadinessError("reviewed action facts cannot replace their own pin")
    pins = {pin.path: pin.sha256, **sources}
    for name, digest in pins.items():
        memory_guard()
        source = inside(root, name)
        if source.stat().st_size > 16 * 1024**2 or file_sha256(source) != digest:
            raise DataReadinessError("reviewed action source changed")
    verified_receipts: set[str] = set()
    for receipt_pin in receipts:
        receipt_path = inside(root, receipt_pin.path)
        if (pins.get(receipt_pin.path) != receipt_pin.sha256 or receipt_path.name != "receipt.json"
                or receipt_path.stat().st_size > 256 * 1024):
            raise DataReadinessError("reviewed action facts lack their exact official receipt")
        if receipt_pin.path in verified_receipts:
            continue
        receipt = parse_strict_json_object(receipt_path.read_bytes(), label="reviewed official receipt")
        request_path, transport_path = receipt_path.parent / "_request.json", receipt_path.parent / "transport.json"
        request_name, transport_name = request_path.relative_to(root).as_posix(), transport_path.relative_to(root).as_posix()
        if (request_name not in pins or transport_name not in pins
                or request_path.stat().st_size > 256 * 1024 or transport_path.stat().st_size > 256 * 1024):
            raise DataReadinessError("reviewed official request or transport missing")
        request = parse_strict_json_object(request_path.read_bytes(), label="reviewed official request")
        transport = parse_strict_json_object(transport_path.read_bytes(), label="reviewed official transport")
        body_leaf = receipt.get("body_path")
        if not isinstance(body_leaf, str):
            raise DataReadinessError("reviewed official receipt body path is missing")
        body_path = inside(receipt_path.parent, body_leaf)
        body_name = body_path.relative_to(root).as_posix()
        if (receipt.get("state") != "archived_unreviewed" or receipt.get("status_code") != 200
                or transport.get("status_code") != 200 or transport.get("redirect_chain")
                or receipt.get("request_sha256") != json_sha256(request)
                or request.get("url") != receipt.get("url")
                or transport.get("requested_url") != request.get("url")
                or transport.get("final_url") != request.get("url")
                or pins.get(body_name) != receipt.get("body_sha256")
                or transport.get("sha256") != receipt.get("body_sha256")
                or pins.get(transport_name) != receipt.get("transport_sha256")
                or transport.get("body_length") != body_path.stat().st_size):
            raise DataReadinessError("reviewed official source is not the exact successful response")
        verified_receipts.add(receipt_pin.path)
    return pins
