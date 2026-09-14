"""Restore lossy compact identity clocks only from pinned original UTC evidence."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256

_CLOCK = "identity_available_at_utc"
_KEYS = ("relation_id", "event_id", "target_security_id", "target_ticker")


def restore_compact_identity_clocks(frame: pd.DataFrame, children: Mapping[str, Mapping[str, str]]) -> pd.DataFrame:
    """Keep source instants exact; naive clocks without original proof remain errors.

    Compact preparation preserves relation IDs, and its child-set hash binds each
    row to its original collection/attribution/sentiment files. No ticker heuristic
    or timezone assumption is used to recover the serialization loss.
    """
    if len(frame) > 250_000 or int(frame.memory_usage(deep=True).sum()) > 128 * 1024**2:
        raise DataReadinessError("compact clock recovery exceeds bounded frame size")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise DataReadinessError("compact clock recovery requires unique row and column indices")
    naive = frame[_CLOCK].map(lambda value: False if pd.isna(value) else pd.Timestamp(value).tzinfo is None)
    if not naive.any():
        return frame.copy()
    required = {*_KEYS, "preparation_original_chunk_id", "preparation_source_child_set_sha256"}
    if not required.issubset(frame.columns) or "compact_original_identity_available_at_utc" in frame:
        raise DataReadinessError("compact clock recovery lacks original row provenance")
    output = frame.copy()
    output["compact_original_identity_available_at_utc"] = frame[_CLOCK]
    output["identity_clock_restoration_source_sha256"] = ""
    restored = list(frame[_CLOCK])
    positions = {label: position for position, label in enumerate(frame.index)}
    for chunk, group in frame.loc[naive].groupby("preparation_original_chunk_id", sort=False, dropna=False):
        pins = children.get(chunk)
        if not pins or not group.preparation_source_child_set_sha256.eq(json_sha256(dict(pins))).all():
            raise DataReadinessError("compact identity clock original child-set proof differs")
        paths = [Path(name) for name in pins if Path(name).suffix == ".parquet"
            and Path(name).parent.name == "relations" and Path(name).parent.parent.name == "attribution"]
        if len(paths) != 1:
            raise DataReadinessError("compact identity clock requires exactly one original relation artifact")
        path = paths[0]
        for source in (path, manifest_path_for(path)):
            digest = pins.get(str(source))
            if digest is None or file_sha256(source) != digest:
                raise DataReadinessError("compact identity clock original artifact pin differs")
        original, _ = load_canonical_artifact(path, expected_type="event_security_relations", allow_research=True,
            columns=[*_KEYS, _CLOCK])
        if len(original) > 250_000 or original.relation_id.duplicated().any():
            raise DataReadinessError("compact identity clock original relation identity is ambiguous")
        original = original.set_index("relation_id", drop=False)
        for label, row in group.iterrows():
            if row.relation_id not in original.index:
                raise DataReadinessError("compact identity clock original relation is missing")
            source_row = original.loc[row.relation_id]
            if any(source_row[key] != row[key] for key in _KEYS):
                raise DataReadinessError("compact identity clock original relation keys differ")
            source_clock = source_row[_CLOCK]
            if pd.isna(source_clock):
                raise DataReadinessError("compact identity clock original timestamp is missing")
            stamp = pd.Timestamp(source_clock)
            if (stamp.tzinfo is None or stamp.utcoffset() != timedelta(0)
                    or stamp.tz_localize(None) != pd.Timestamp(row[_CLOCK])):
                raise DataReadinessError("compact identity clock lacks exact aware UTC source proof")
            restored[positions[label]] = stamp
            output.loc[label, "identity_clock_restoration_source_sha256"] = pins[str(path)]
        for source in (path, manifest_path_for(path)):
            if file_sha256(source) != pins[str(source)]:
                raise DataReadinessError("compact identity clock original artifact pin changed during read")
    for value in restored:
        if not pd.isna(value):
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
                raise DataReadinessError("compact identity clocks require aware UTC before normalization")
    output[_CLOCK] = pd.Series(restored, index=frame.index, dtype="datetime64[ns, UTC]")
    return output
