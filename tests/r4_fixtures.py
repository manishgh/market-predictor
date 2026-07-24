import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.evaluation import session_block_interval
from scripts.promotion_fixture import (
    authorize_candidate_for_test as authorize_candidate_for_test,
)
from scripts.promotion_fixture import (
    synthetic_identity_metrics as synthetic_identity_metrics,
)
from scripts.promotion_fixture import (
    test_signing_material as test_signing_material,
)
from scripts.promotion_fixture import (
    trust_context_for_candidate as trust_context_for_candidate,
)


def write_test_shadow_bundle(
    root: Path,
    sessions: pd.DataFrame,
    *,
    hypothesis: dict[str, Any],
    candidate_artifact_sha256: str,
    generated_at: datetime,
    bootstrap_iterations: int,
    bootstrap_seed: int = 42,
) -> Path:
    evidence = sessions.copy()
    evidence["session_date_et"] = pd.to_datetime(
        evidence["session_date_et"]
    ).dt.date
    evidence["paired_improvement"] = (
        evidence["candidate_benchmark_excess_return"]
        - evidence["baseline_benchmark_excess_return"]
    )
    interval = session_block_interval(
        evidence,
        metric=lambda frame: float(frame["paired_improvement"].mean()),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    records = [
        {
            "session_date_et": cast(date, row.session_date_et).isoformat(),
            "candidate_benchmark_excess_return": float(
                row.candidate_benchmark_excess_return
            ),
            "baseline_benchmark_excess_return": float(
                row.baseline_benchmark_excess_return
            ),
        }
        for row in evidence.itertuples(index=False)
    ]
    declared = datetime.fromisoformat(str(hypothesis["declared_at_utc"]))
    if date.fromisoformat(records[0]["session_date_et"]) <= declared.date():
        raise DataReadinessError(
            "every shadow session must follow hypothesis declaration"
        )
    content: dict[str, Any] = {
        "schema": "market_predictor.test_shadow_evidence.v1",
        "hypothesis_id": hypothesis["hypothesis_id"],
        "hypothesis_family": hypothesis["hypothesis_family"],
        "hypothesis_record_sha256": hypothesis["record_sha256"],
        "baseline_id": hypothesis["baseline_id"],
        "baseline_artifact_sha256": hypothesis[
            "baseline_artifact_sha256"
        ],
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "prediction_policy_sha256": hypothesis[
            "prediction_policy_sha256"
        ],
        "execution_policy_sha256": hypothesis[
            "execution_policy_sha256"
        ],
        "source_evidence_sha256": _sha(records),
        "generated_at_utc": generated_at.isoformat(),
        "first_session_date_et": records[0]["session_date_et"],
        "last_session_date_et": records[-1]["session_date_et"],
        "independent_sessions": len(records),
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
        "paired_improvement_interval": interval,
        "session_returns": records,
    }
    payload = {**content, "shadow_fingerprint": _sha(content)}
    path = root / "shadow" / f"{payload['shadow_fingerprint']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def load_test_shadow_bundle(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("test shadow bundle must contain an object")
    return {str(key): value for key, value in loaded.items()}


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
