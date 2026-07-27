from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
from unittest.mock import Mock, patch

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday import specialist_experiments
from market_predictor.intraday.specialist_contracts import (
    INTRADAY_SPECIALIST_IDS,
    IntradaySpecialistResearchConfig,
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_model import (
    SPECIALIST_ACCEPTED_STATUS,
    SPECIALIST_REJECTED_STATUS,
    RetainedSpecialistModel,
    SpecialistExperimentResult,
    SpecialistExperimentSpec,
    SpecialistSplitPlan,
)
from market_predictor.intraday.specialist_training_data import (
    SPECIALIST_TRAINING_DATASET_SCHEMA,
    SPECIALIST_TRAINING_ROW_SCHEMA,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ID = "INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1"


class IntradaySpecialistExperimentTests(unittest.TestCase):
    config: ClassVar[IntradaySpecialistResearchConfig]

    @classmethod
    def setUpClass(cls) -> None:
        base = load_intraday_specialist_research_config(
            ROOT / "configs" / "intraday_specialist_research.toml"
        )
        cls.config = base.model_copy(
            update={
                "n_splits": 3,
                "min_train_sessions": 12,
                "min_train_rows": 80,
                "min_training_tickers": 8,
                "ticker_holdout_fraction": 0.20,
                "top_k": 2,
                "max_trades_per_session": 2,
                "minimum_selected_trades": 1,
                "minimum_avg_net_return": -1.0,
                "minimum_avg_excess_return_vs_spy": -1.0,
                "minimum_avg_excess_return_vs_sector": -1.0,
                "minimum_avg_net_return_ci_low": -1.0,
                "minimum_avg_excess_return_vs_spy_ci_low": -1.0,
                "minimum_profit_factor": 0.0,
                "maximum_drawdown": 1.0,
                "maximum_negative_session_rate": 1.0,
                "required_market_regimes": ("risk_on", "risk_off"),
                "minimum_regime_selected_trades": 1,
                "minimum_regime_avg_net_return": -1.0,
                "minimum_regime_avg_excess_return_vs_spy": -1.0,
                "technical_features": (
                    "atr_pct",
                    "volatility_12bar",
                    "volatility_6bar",
                    "return_1bar",
                    "rel_return_3bar_vs_sector",
                    "close_location_5m",
                    "dist_opening_range_high",
                    "relative_volume_same_minute_20d",
                    "rel_return_3bar_vs_qqq",
                ),
            }
        )

    def test_writes_atomic_accepted_and_rejected_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_training_bundle(
                Path(directory), self.config
            )
            out_dir = Path(directory) / "experiments"
            evaluator = Mock(side_effect=_fake_result)
            memory = Mock()
            with (
                patch.object(
                    specialist_experiments,
                    "evaluate_specialist_experiment",
                    evaluator,
                ),
                patch.object(
                    specialist_experiments,
                    "assert_memory_budget",
                    memory,
                ),
                patch.object(
                    specialist_experiments,
                    "assert_peak_memory_budget",
                ),
            ):
                result = specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                )

            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(evaluator.call_count, 3)
            self.assertGreaterEqual(memory.call_count, 4)
            strategy_dir = (
                out_dir
                / "strategies"
                / specialist_experiments._slug(STRATEGY_ID)
            )
            manifest = _json(strategy_dir / "_manifest.json")
            self.assertEqual(
                manifest["accepted_development_count"], 1
            )
            self.assertEqual(manifest["rejected_count"], 2)
            candidates = strategy_dir / "candidates"
            accepted = (
                candidates
                / specialist_experiments._slug(
                    "deterministic_baseline"
                )
            )
            self.assertTrue((accepted / "model.joblib").is_file())
            for candidate_id in (
                "logistic",
                "hist_gradient_boosting",
            ):
                candidate = (
                    candidates
                    / specialist_experiments._slug(candidate_id)
                )
                self.assertFalse((candidate / "model.joblib").exists())
                self.assertTrue(
                    (candidate / "predictions.parquet").is_file()
                )
                authority = _json(candidate / "_authority.json")
                self.assertEqual(authority["state"], "complete")
                self.assertEqual(
                    authority["artifact_sha256"],
                    file_sha256(candidate / "_manifest.json"),
                )
            root_request = _json(out_dir / "_request.json")
            catalyst = specialist_experiments._mapping(
                root_request["catalyst_overlay"]
            )
            implementation = specialist_experiments._mapping(
                root_request["implementation"]
            )
            self.assertEqual(
                root_request["training_dataset_fingerprint"],
                fixture.dataset_fingerprint,
            )
            self.assertEqual(
                catalyst["status"],
                "data_blocked",
            )
            self.assertEqual(
                len(
                    str(implementation["implementation_sha256"])
                ),
                64,
            )
            root_authority = _json(out_dir / "_authority.json")
            self.assertEqual(root_authority["state"], "incomplete")
            self.assertFalse(
                any(out_dir.rglob("*.tmp"))
            )

    def test_verified_candidates_resume_without_evaluation(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_training_bundle(
                Path(directory), self.config
            )
            out_dir = Path(directory) / "experiments"
            with patch.object(
                specialist_experiments,
                "evaluate_specialist_experiment",
                side_effect=_fake_result,
            ):
                first = specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                )
            progress: list[object] = []
            with patch.object(
                specialist_experiments,
                "evaluate_specialist_experiment",
                side_effect=AssertionError("candidate was not resumed"),
            ) as evaluator:
                second = specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                    progress=progress.append,
                )

            self.assertEqual(first, second)
            evaluator.assert_not_called()

    def test_tampered_candidate_is_not_resumed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_training_bundle(
                Path(directory), self.config
            )
            out_dir = Path(directory) / "experiments"
            with patch.object(
                specialist_experiments,
                "evaluate_specialist_experiment",
                side_effect=_fake_result,
            ):
                specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                )
            prediction_path = (
                out_dir
                / "strategies"
                / specialist_experiments._slug(STRATEGY_ID)
                / "candidates"
                / "deterministic_baseline"
                / "predictions.parquet"
            )
            prediction_path.write_bytes(
                prediction_path.read_bytes() + b"tamper"
            )

            with self.assertRaisesRegex(
                DataReadinessError, "does not verify"
            ):
                specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                )

    def test_training_shard_hash_and_fingerprint_are_verified(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _write_training_bundle(root, self.config)
            shard = next(
                (fixture.dataset_dir / "strategies").rglob("*.parquet")
            )
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                DataReadinessError, "file record"
            ):
                specialist_experiments.verify_intraday_specialist_training_bundle(
                    fixture.dataset_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _write_training_bundle(root, self.config)
            manifest_path = fixture.dataset_dir / "_manifest.json"
            manifest = _json(manifest_path)
            manifest["dataset_fingerprint"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                DataReadinessError, "fingerprint"
            ):
                specialist_experiments.verify_intraday_specialist_training_bundle(
                    fixture.dataset_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                )

    def test_accepted_status_without_model_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_training_bundle(
                Path(directory), self.config
            )
            out_dir = Path(directory) / "experiments"

            def invalid_result(
                plan: SpecialistSplitPlan,
                spec: SpecialistExperimentSpec,
                *,
                config: IntradaySpecialistResearchConfig,
            ) -> SpecialistExperimentResult:
                result = _fake_result(plan, spec, config=config)
                return SpecialistExperimentResult(
                    spec=result.spec,
                    status=SPECIALIST_ACCEPTED_STATUS,
                    rejection_reasons=(),
                    metrics=result.metrics,
                    predictions=result.predictions,
                    economics=result.economics,
                    regime_evidence=result.regime_evidence,
                    fold_audit=result.fold_audit,
                    retained_model=None,
                )

            with patch.object(
                specialist_experiments,
                "evaluate_specialist_experiment",
                side_effect=invalid_result,
            ):
                result = specialist_experiments.train_intraday_specialist_experiments(
                    dataset_dir=fixture.dataset_dir,
                    out_dir=out_dir,
                    config=self.config,
                    policy_path=fixture.policy_path,
                    strategy_ids=[STRATEGY_ID],
                )

            failures = specialist_experiments._mapping(
                result["failed_strategies"]
            )
            self.assertIn(STRATEGY_ID, failures)
            strategy_dir = (
                out_dir
                / "strategies"
                / specialist_experiments._slug(STRATEGY_ID)
            )
            self.assertFalse(
                any(strategy_dir.rglob("model.joblib"))
            )


class _Fixture:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        policy_path: Path,
        dataset_fingerprint: str,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.policy_path = policy_path
        self.dataset_fingerprint = dataset_fingerprint


def _write_training_bundle(
    root: Path,
    config: IntradaySpecialistResearchConfig,
) -> _Fixture:
    selected_training = _training_rows()
    policy_path = root / "configs" / "intraday_specialist_research.toml"
    policy_path.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "configs" / "intraday_specialist_research.toml",
        policy_path,
    )
    setup_dir = root / "data" / "setup"
    plan_dir = root / "data" / "plan"
    collection_dir = root / "data" / "collection"
    setup_dir.mkdir(parents=True)
    plan_dir.mkdir(parents=True)
    collection_dir.mkdir(parents=True)
    setup_fingerprint = "1" * 64
    plan_fingerprint = "2" * 64
    collection_request = "3" * 64
    _write_json(
        setup_dir / "_manifest.json",
        {
            "schema": "intraday.specialist_setup_bundle.v1",
            "bundle_fingerprint": setup_fingerprint,
        },
    )
    _write_json(
        plan_dir / "_manifest.json",
        {
            "schema": "intraday.specialist_collection_plan.v1",
            "plan_fingerprint": plan_fingerprint,
        },
    )
    _write_json(
        collection_dir / "_manifest.json",
        {
            "schema": "intraday.specialist_one_minute_collection.v1",
            "status": "transport_complete",
            "request_sha256": collection_request,
            "failed_units": {},
            "completed_units": 1,
            "requested_units": 1,
        },
    )
    dataset_dir = root / "data" / "training"
    files: list[dict[str, object]] = []
    strategy_rows: dict[str, int] = {}
    for strategy_id in INTRADAY_SPECIALIST_IDS:
        strategy_dir = (
            dataset_dir
            / "strategies"
            / specialist_experiments._slug(strategy_id)
        )
        strategy_dir.mkdir(parents=True, exist_ok=True)
        path = strategy_dir / "2025-01-part-000.parquet"
        frame = (
            selected_training
            if strategy_id == STRATEGY_ID
            else pd.DataFrame(
                {
                    "training_schema_version": [
                        SPECIALIST_TRAINING_ROW_SCHEMA
                    ],
                    "strategy_id": [strategy_id],
                }
            )
        )
        frame.to_parquet(path, index=False)
        record = {
            "path": path.relative_to(dataset_dir).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
        files.append(record)
        strategy_rows[strategy_id] = len(frame)
    collection_manifest_sha256 = file_sha256(
        collection_dir / "_manifest.json"
    )
    fingerprint = specialist_experiments._training_dataset_fingerprint(
        files=files,
        setup_fingerprint=setup_fingerprint,
        plan_fingerprint=plan_fingerprint,
        collection_manifest_sha256=collection_manifest_sha256,
        policy_sha256=config.policy_sha256(),
    )
    _write_json(
        dataset_dir / "_manifest.json",
        {
            "schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
            "row_schema": SPECIALIST_TRAINING_ROW_SCHEMA,
            "dataset_fingerprint": fingerprint,
            "policy": {
                "path": "configs/intraday_specialist_research.toml",
                "file_sha256": file_sha256(policy_path),
                "policy_sha256": config.policy_sha256(),
                "schema_version": config.schema_version,
            },
            "setup_bundle": {
                "path": "data/setup",
                "manifest_sha256": file_sha256(
                    setup_dir / "_manifest.json"
                ),
                "bundle_fingerprint": setup_fingerprint,
            },
            "collection_plan": {
                "path": "data/plan",
                "manifest_sha256": file_sha256(
                    plan_dir / "_manifest.json"
                ),
                "plan_fingerprint": plan_fingerprint,
            },
            "collection": {
                "path": "data/collection",
                "manifest_sha256": collection_manifest_sha256,
                "request_sha256": collection_request,
                "rows": 1,
            },
            "files": files,
            "summary": {
                "rows": sum(strategy_rows.values()),
                "eligible_rows": len(selected_training),
                "strategy_rows": strategy_rows,
                "strategy_eligible_rows": {
                    strategy_id: (
                        len(selected_training)
                        if strategy_id == STRATEGY_ID
                        else 0
                    )
                    for strategy_id in INTRADAY_SPECIALIST_IDS
                },
                "ineligible_reason_counts": {},
            },
        },
    )
    return _Fixture(
        dataset_dir=dataset_dir,
        policy_path=policy_path,
        dataset_fingerprint=fingerprint,
    )


def _training_rows() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2025-01-02", "2025-03-31")[
        :48
    ]
    tickers = [f"T{index:02d}" for index in range(10)]
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        session_date = pd.Timestamp(session).date()
        decision = pd.Timestamp(
            calendar.session_open(session)
        ).tz_convert("UTC") + pd.Timedelta(minutes=31)
        window_end = decision + pd.Timedelta(minutes=60)
        for ticker_index, ticker in enumerate(tickers):
            opportunity = int(
                (session_index + ticker_index) % 3 != 0
            )
            downside = 1 - opportunity
            net = 0.01 if opportunity else -0.006
            signal = (
                0.02 if opportunity else -0.01
            ) + ticker_index / 10_000
            rows.append(
                {
                    "training_schema_version": (
                        SPECIALIST_TRAINING_ROW_SCHEMA
                    ),
                    "strategy_id": STRATEGY_ID,
                    "setup_id": f"{session_date}:{ticker}",
                    "ticker": ticker,
                    "session_date_et": session_date,
                    "decision_time_utc": decision,
                    "feature_available_at_utc": (
                        decision - pd.Timedelta(seconds=30)
                    ),
                    "entry_time_utc": decision,
                    "exit_time_utc": window_end,
                    "label_available_at_utc": (
                        window_end + pd.Timedelta(seconds=30)
                    ),
                    "label_window_end_utc": window_end,
                    "label_eligible": True,
                    "horizon_minutes": 60,
                    "path_outcome": (
                        "target" if opportunity else "stop"
                    ),
                    "target_before_stop_60m": opportunity,
                    "stop_before_target_60m": downside,
                    "path_realized_return_gross_60m": net + 0.001,
                    "path_realized_return_net_60m": net,
                    "path_spy_return_60m": 0.0,
                    "path_qqq_return_60m": 0.0,
                    "path_sector_return_60m": 0.0,
                    "path_excess_return_60m_vs_spy": net,
                    "path_excess_return_60m_vs_qqq": net,
                    "path_excess_return_60m_vs_sector": net,
                    "entry_price": 20.0,
                    "entry_dollar_volume": 2_000_000.0,
                    "entry_atr_pct": 0.02,
                    "sector": (
                        "Technology"
                        if ticker_index % 2
                        else "Healthcare"
                    ),
                    "primary_benchmark": (
                        "XLK" if ticker_index % 2 else "XLV"
                    ),
                    "market_cap_bucket": (
                        "mid" if ticker_index % 2 else "small"
                    ),
                    "liquidity_bucket": (
                        "high" if ticker_index % 2 else "medium"
                    ),
                    "regime_risk_on": int(session_index % 3 == 0),
                    "regime_risk_off": int(session_index % 3 == 1),
                    "regime_high_volatility": 0,
                    "atr_pct": 0.02 + downside * 0.01,
                    "volatility_12bar": 0.01 + downside * 0.01,
                    "volatility_6bar": 0.01 + downside * 0.01,
                    "return_1bar": signal,
                    "rel_return_3bar_vs_sector": signal,
                    "close_location_5m": (
                        0.8 if opportunity else 0.55
                    ),
                    "dist_opening_range_high": max(signal, 0),
                    "relative_volume_same_minute_20d": (
                        2.0 if opportunity else 1.3
                    ),
                    "rel_return_3bar_vs_qqq": signal,
                }
            )
    return pd.DataFrame(rows)


def _fake_result(
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    *,
    config: IntradaySpecialistResearchConfig,
) -> SpecialistExperimentResult:
    del config
    accepted = spec.estimator_family == "deterministic_baseline"
    status = (
        SPECIALIST_ACCEPTED_STATUS
        if accepted
        else SPECIALIST_REJECTED_STATUS
    )
    retained = (
        RetainedSpecialistModel(
            estimators={
                plan.opportunity_target: None,
                plan.downside_target: None,
            },
            calibrators={
                plan.opportunity_target: "opportunity-calibrator",
                plan.downside_target: "downside-calibrator",
            },
            features=plan.features,
            opportunity_target=plan.opportunity_target,
            downside_target=plan.downside_target,
        )
        if accepted
        else None
    )
    return SpecialistExperimentResult(
        spec=spec,
        status=status,
        rejection_reasons=() if accepted else ("economic gate",),
        metrics={
            "split_sha256": plan.split_sha256,
            "features": list(plan.features),
            "status": status,
        },
        predictions=pd.DataFrame(
            {
                "ticker": ["AAA"],
                "selection_raw_score": [0.5],
            }
        ),
        economics=pd.DataFrame(
            {"scope": ["walk_forward"], "selected_trades": [1]}
        ),
        regime_evidence=pd.DataFrame(
            {"scope": ["regime:risk_on:walk_forward"]}
        ),
        fold_audit=pd.DataFrame(
            {"record_type": ["validation_fold"]}
        ),
        retained_model=retained,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


if __name__ == "__main__":
    unittest.main()
