"""Blind review, adjudication, and resolution for issuer-event precision evidence."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.catalysts.issuer_events.classification import (
    EVENT_FAMILIES,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.issuer_event_precision.artifact_integrity import (
    _assert_frame_equal,
    _audit_report,
    _clean_text,
    _read_csv,
)
from market_predictor.governance.issuer_event_precision.contracts import (
    _CORRECTION_FIELDS,
    _DECISION_FIELDS,
    _YES_NO,
    _YES_NO_UNCERTAIN,
    ADJUDICATION_TEMPLATE_COLUMNS,
    FINAL_MANIFEST_SCHEMA,
    REVIEW_COLUMNS,
    REVIEW_TEMPLATE_COLUMNS,
)


def _preflight_review_ledgers(
    *,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
) -> None:
    """Reject malformed review logic before the expensive authority replay."""

    reviewer_one = _read_csv(reviewer_one_path, REVIEW_TEMPLATE_COLUMNS)
    reviewer_two = _read_csv(reviewer_two_path, REVIEW_TEMPLATE_COLUMNS)
    adjudication = _read_csv(adjudication_path, ADJUDICATION_TEMPLATE_COLUMNS)
    identity_columns = ("sample_id", "family_event_id", "event_family")
    _assert_frame_equal(
        reviewer_one.loc[:, identity_columns],
        reviewer_two.loc[:, identity_columns],
        "reviewer preflight identity",
    )
    _assert_frame_equal(
        reviewer_one.loc[:, identity_columns],
        adjudication.loc[:, identity_columns],
        "adjudication preflight identity",
    )
    sample = pd.DataFrame(
        {
            "sample_id": reviewer_one["sample_id"].astype(str),
            "family_event_id": reviewer_one["family_event_id"].astype(str),
            "proposed_event_family": reviewer_one["event_family"].astype(str),
            "sample_role": "preflight",
            "inference_cluster_id": reviewer_one["sample_id"].astype(str),
        }
    )
    normalized_one = _load_review_ledger(
        reviewer_one_path,
        sample,
        reviewer_slot=1,
    )
    normalized_two = _load_review_ledger(
        reviewer_two_path,
        sample,
        reviewer_slot=2,
    )
    normalized_adjudication = _load_adjudication_ledger(adjudication_path, sample)
    _resolve_reviews(
        sample,
        normalized_one,
        normalized_two,
        normalized_adjudication,
    )

def _review_template(sample: pd.DataFrame, *, reviewer_slot: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": sample["sample_id"].astype(str),
            "family_event_id": sample["family_event_id"].astype(str),
            "event_family": sample["proposed_event_family"].astype(str),
            "reviewer_slot": str(reviewer_slot),
            "reviewer_id": "",
            "family_correct": "",
            "issuer_target_correct": "",
            "event_announced_or_completed": "",
            "correct_family": "",
            "action_subject_text": "",
            "false_positive_reason": "",
            "comments": "",
        }
    )
    return frame.loc[:, REVIEW_TEMPLATE_COLUMNS]


def _adjudication_template(sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": sample["sample_id"].astype(str),
            "family_event_id": sample["family_event_id"].astype(str),
            "event_family": sample["proposed_event_family"].astype(str),
            "adjudicator_id": "",
            "family_correct": "",
            "issuer_target_correct": "",
            "event_announced_or_completed": "",
            "correct_family": "",
            "action_subject_text": "",
            "false_positive_reason": "",
            "comments": "",
        }
    )
    return frame.loc[:, ADJUDICATION_TEMPLATE_COLUMNS]


def _load_review_ledger(
    path: Path,
    sample: pd.DataFrame,
    *,
    reviewer_slot: int,
) -> pd.DataFrame:
    frame = _read_csv(path, REVIEW_TEMPLATE_COLUMNS)
    expected = _review_template(sample, reviewer_slot=reviewer_slot)
    _validate_ledger_identity(frame, expected, reviewer_slot=reviewer_slot)
    frame["reviewer_id"] = frame["reviewer_id"].map(_normalized_review_text)
    if bool(frame["reviewer_id"].eq("").any()):
        raise DataReadinessError("precision reviewer identity is incomplete")
    if frame["reviewer_id"].nunique(dropna=False) != 1:
        raise DataReadinessError(f"precision reviewer slot {reviewer_slot} must use exactly one identity")
    for column in _DECISION_FIELDS:
        normalized = frame[column].str.strip().str.lower()
        if not set(normalized).issubset(_YES_NO_UNCERTAIN):
            raise DataReadinessError(f"precision review has invalid {column}")
        frame[column] = normalized
    for column in (*_CORRECTION_FIELDS, "comments"):
        frame[column] = frame[column].str.strip()
    frame["correct_family"] = frame["correct_family"].str.lower()
    _validate_correction_fields(frame, context=f"reviewer {reviewer_slot}")
    return frame


def _load_adjudication_ledger(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(path, ADJUDICATION_TEMPLATE_COLUMNS)
    expected = _adjudication_template(sample)
    identity_columns = ("sample_id", "family_event_id", "event_family")
    _assert_frame_equal(
        frame.loc[:, identity_columns],
        expected.loc[:, identity_columns],
        "adjudication identity",
    )
    if bool(frame["sample_id"].duplicated().any()):
        raise DataReadinessError("precision adjudication contains duplicate samples")
    for column in ("adjudicator_id", *_DECISION_FIELDS, *_CORRECTION_FIELDS, "comments"):
        frame[column] = frame[column].str.strip()
    frame["adjudicator_id"] = frame["adjudicator_id"].map(_normalized_review_text)
    adjudicators = frame.loc[frame["adjudicator_id"].ne(""), "adjudicator_id"]
    if adjudicators.nunique(dropna=False) > 1:
        raise DataReadinessError("precision adjudication slot must use at most one identity")
    frame["correct_family"] = frame["correct_family"].str.lower()
    return frame


def _validate_ledger_identity(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    reviewer_slot: int,
) -> None:
    identity_columns = (
        "sample_id",
        "family_event_id",
        "event_family",
        "reviewer_slot",
    )
    if not frame["reviewer_slot"].eq(str(reviewer_slot)).all():
        raise DataReadinessError("precision reviewer slot differs from its template")
    _assert_frame_equal(
        frame.loc[:, identity_columns].astype(str),
        expected.loc[:, identity_columns].astype(str),
        f"reviewer {reviewer_slot} identity",
    )
    if bool(frame["sample_id"].duplicated().any()):
        raise DataReadinessError("precision review contains duplicate samples")


def _resolve_reviews(
    sample: pd.DataFrame,
    reviewer_one: pd.DataFrame,
    reviewer_two: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> pd.DataFrame:
    one = reviewer_one.set_index("sample_id", drop=False)
    two = reviewer_two.set_index("sample_id", drop=False)
    adjudicated = adjudication.set_index("sample_id", drop=False)
    rows: list[dict[str, object]] = []
    for sample_row in sample.to_dict(orient="records"):
        sample_id = str(sample_row["sample_id"])
        first = one.loc[sample_id]
        second = two.loc[sample_id]
        reviewer_one_id = str(first["reviewer_id"]).strip()
        reviewer_two_id = str(second["reviewer_id"]).strip()
        if reviewer_one_id.casefold() == reviewer_two_id.casefold():
            raise DataReadinessError("precision audit cannot use the same reviewer twice")
        disagreement = any(
            str(first[column]) != str(second[column]) or str(first[column]) == "uncertain" or str(second[column]) == "uncertain"
            for column in _DECISION_FIELDS
        ) or any(_normalized_review_text(first[column]) != _normalized_review_text(second[column]) for column in _CORRECTION_FIELDS)
        adjudication_row = adjudicated.loc[sample_id]
        adjudicator_id = str(adjudication_row["adjudicator_id"]).strip()
        final_values: dict[str, bool]
        resolution_state: str
        if disagreement:
            adjudication_values = {column: str(adjudication_row[column]).strip().lower() for column in _DECISION_FIELDS}
            complete = bool(adjudicator_id) and set(adjudication_values.values()).issubset(_YES_NO)
            if complete:
                if adjudicator_id.casefold() in {
                    reviewer_one_id.casefold(),
                    reviewer_two_id.casefold(),
                }:
                    raise DataReadinessError("precision adjudicator must differ from both reviewers")
                final_values = {column: value == "yes" for column, value in adjudication_values.items()}
                _validate_correction_row(
                    adjudication_row,
                    decisions=adjudication_values,
                    context="adjudication",
                )
                resolution_state = "adjudicated"
            else:
                if (
                    adjudicator_id
                    or any(adjudication_values.values())
                    or any(str(adjudication_row[column]).strip() for column in _CORRECTION_FIELDS)
                ):
                    raise DataReadinessError("precision adjudication is partially completed")
                final_values = {column: False for column in _DECISION_FIELDS}
                resolution_state = "unresolved_failure"
        else:
            if adjudicator_id or any(str(adjudication_row[column]).strip() for column in (*_DECISION_FIELDS, *_CORRECTION_FIELDS)):
                raise DataReadinessError("precision adjudication cannot override agreeing reviewers")
            final_values = {column: str(first[column]) == "yes" for column in _DECISION_FIELDS}
            resolution_state = "reviewer_agreement"
        notes_source = adjudication_row if resolution_state == "adjudicated" else first
        rows.append(
            {
                "sample_id": sample_id,
                "sample_role": str(sample_row["sample_role"]),
                "inference_cluster_id": str(sample_row["inference_cluster_id"]),
                "family_event_id": str(sample_row["family_event_id"]),
                "event_family": str(sample_row["proposed_event_family"]),
                "reviewer_one_id": reviewer_one_id,
                "reviewer_two_id": reviewer_two_id,
                "adjudicator_id": adjudicator_id,
                "reviewer_one_family_correct": str(first["family_correct"]),
                "reviewer_one_issuer_target_correct": str(first["issuer_target_correct"]),
                "reviewer_one_event_announced_or_completed": str(first["event_announced_or_completed"]),
                "reviewer_two_family_correct": str(second["family_correct"]),
                "reviewer_two_issuer_target_correct": str(second["issuer_target_correct"]),
                "reviewer_two_event_announced_or_completed": str(second["event_announced_or_completed"]),
                "adjudication_required": disagreement,
                "resolution_state": resolution_state,
                **final_values,
                "joint_correct": all(final_values.values()),
                "wrong_issuer": (resolution_state != "unresolved_failure" and not final_values["issuer_target_correct"]),
                "correct_family": str(notes_source["correct_family"]).strip(),
                "action_subject_text": str(notes_source["action_subject_text"]).strip(),
                "false_positive_reason": str(notes_source["false_positive_reason"]).strip(),
                "comments": str(notes_source["comments"]).strip(),
                "schema_version": FINAL_MANIFEST_SCHEMA,
            }
        )
    output = pd.DataFrame.from_records(rows, columns=REVIEW_COLUMNS)
    _review_audit(output, sample).raise_for_failure()
    return output


def _validate_correction_fields(frame: pd.DataFrame, *, context: str) -> None:
    for row in frame.to_dict(orient="records"):
        _validate_correction_row(
            row,
            decisions={field: str(row[field]) for field in _DECISION_FIELDS},
            context=context,
        )


def _validate_correction_row(
    row: Mapping[str, object] | pd.Series,
    *,
    decisions: Mapping[str, str],
    context: str,
) -> None:
    correct_family = _clean_text(row["correct_family"])
    action_subject = _clean_text(row["action_subject_text"])
    false_positive_reason = _clean_text(row["false_positive_reason"])
    if decisions["family_correct"] == "no" and correct_family not in {
        *EVENT_FAMILIES,
        "none",
    }:
        raise DataReadinessError(f"{context} must supply a valid correct_family for an incorrect family")
    if decisions["family_correct"] == "yes" and correct_family:
        raise DataReadinessError(f"{context} cannot supply correct_family when the family is correct")
    if decisions["issuer_target_correct"] == "no" and not action_subject:
        raise DataReadinessError(f"{context} must identify the action subject for a wrong issuer")
    if decisions["issuer_target_correct"] == "yes" and action_subject:
        raise DataReadinessError(f"{context} cannot supply an action subject when the issuer is correct")
    if decisions["event_announced_or_completed"] == "no" and not false_positive_reason:
        raise DataReadinessError(f"{context} must explain a non-event false positive")
    if decisions["event_announced_or_completed"] == "yes" and false_positive_reason:
        raise DataReadinessError(f"{context} cannot supply a false-positive reason for a confirmed event")


def _normalized_review_text(value: object) -> str:
    return " ".join(_clean_text(value).casefold().split())

def _review_audit(frame: pd.DataFrame, sample: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(list(frame.columns) != list(REVIEW_COLUMNS))
    failures += abs(len(frame) - len(sample))
    if not frame.empty:
        failures += int(frame["sample_id"].astype(str).duplicated().sum())
        failures += len(set(frame["sample_id"].astype(str)).symmetric_difference(set(sample["sample_id"].astype(str))))
    return _audit_report("issuer_event_precision_reviews", len(frame), failures)
