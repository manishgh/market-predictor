from __future__ import annotations

import pytest

from market_predictor.core.json_integrity import parse_strict_json_object


def test_strict_json_accepts_finite_nested_object() -> None:
    assert parse_strict_json_object(
        b'{"count":2,"values":[0.1,{"ok":true}]}',
        label="artifact",
    ) == {"count": 2, "values": [0.1, {"ok": True}]}


@pytest.mark.parametrize(
    "payload",
    (
        '{"value":1,"value":2}',
        '{"nested":{"value":1,"value":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e999}',
        "[]",
    ),
)
def test_strict_json_rejects_ambiguous_or_non_finite_payloads(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_strict_json_object(payload, label="artifact")
