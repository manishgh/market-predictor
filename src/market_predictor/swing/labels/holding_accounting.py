"""Model target projection from the same event-aware lot replay used by funding."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_accounting import HoldingSpecification
from market_predictor.swing.evaluation.holding_accounting import project_holding_targets, replay_holding


def build_event_aware_swing_target_row(
    fixed: HoldingSpecification, managed: HoldingSpecification, *,
    benchmarks: Mapping[str, HoldingSpecification], research_contract_sha256: str,
) -> dict[str, Any]:
    """Emit nullable ten-session targets; arithmetic does not admit source evidence."""
    if set(benchmarks) != {"spy", "qqq", "sector"}:
        raise DataReadinessError("event-aware targets require SPY, QQQ and the point-in-time sector benchmark")
    if benchmarks["spy"].security_id != "SPY" or benchmarks["qqq"].security_id != "QQQ":
        raise DataReadinessError("SPY/QQQ target roles require their canonical benchmark identities")
    specifications = (fixed, managed, *(benchmarks[name] for name in ("spy", "qqq", "sector")))
    if any(spec.research_contract_sha256 != research_contract_sha256 for spec in specifications):
        raise DataReadinessError("event-aware target research identities differ")
    fixed_outcome, managed_outcome, *benchmark_outcomes = tuple(replay_holding(spec) for spec in specifications)
    target = project_holding_targets(fixed_outcome, managed_outcome, benchmarks=tuple(benchmark_outcomes))
    row: dict[str, Any] = {
        "decision_id": fixed.decision_id, "security_id": fixed.security_id,
        "entry_time_utc": target.initial_entry_timestamp,
        "exit_time_utc": target.horizon_end_timestamp,
        "managed_exit_time_utc": target.managed_exit_timestamp,
        "label_available_at_utc": target.label_available_at,
        "future_gross_return_10d": target.fixed_horizon_gross_return,
        "future_net_return_10d": target.fixed_horizon_net_return,
        "managed_horizon_gross_return_10d": target.managed_horizon_gross_return,
        "managed_horizon_net_return_10d": target.managed_horizon_net_return,
        "research_contract_sha256": research_contract_sha256,
        "label_eligible": False,
        "source_admission_status": "independent_source_replay_required",
        "fully_settled_at_horizon": fixed_outcome.snapshots[9].fully_settled,
        "managed_fully_settled_at_horizon": managed_outcome.snapshots[9].fully_settled,
        "fixed_horizon_gaps": [gap.model_dump(mode="json") for gap in fixed_outcome.snapshots[9].gaps],
        "managed_horizon_gaps": [gap.model_dump(mode="json") for gap in managed_outcome.snapshots[9].gaps],
    }
    for name, comparison in zip(("spy", "qqq", "sector"), target.benchmarks, strict=True):
        row[f"future_{name}_return_10d"] = comparison.horizon_gross_return
        row[f"future_excess_return_10d_vs_{name}"] = comparison.fixed_horizon_excess_return
        row[f"managed_horizon_excess_return_10d_vs_{name}"] = comparison.managed_horizon_excess_return
        row[f"{name}_security_id"] = comparison.benchmark_security_id
    row["label_values_available"] = all(row[key] is not None for key in (
        "future_net_return_10d", "managed_horizon_net_return_10d",
        "future_excess_return_10d_vs_spy", "future_excess_return_10d_vs_qqq",
        "future_excess_return_10d_vs_sector",
    ))
    return row
