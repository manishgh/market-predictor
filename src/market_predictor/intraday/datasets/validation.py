"""Atomic, lineage-bound publisher for the causal intraday training dataset."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.one_minute_coverage import (
    load_complete_one_minute_coverage,
    verify_canonical_five_minute_store,
)
from market_predictor.intraday.contracts.dataset_schemas import (
    _REQUIRED_BENCHMARKS,
    MAXIMUM_SECURITY_EXCLUSION_FRACTION,
    _VerifiedInputs,
)
from market_predictor.intraday.contracts.history_collection import (
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
)
from market_predictor.intraday.contracts.lineage import (
    DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
    require_intraday_contract_lineage,
)
from market_predictor.intraday.datasets.history import (
    load_complete_intraday_history_plan,
    load_plan_json,
)
from market_predictor.intraday.datasets.history_collection import (
    load_complete_intraday_history_collection,
)
from market_predictor.intraday.datasets.io import (
    _PARQUET_FILE,
    _existing_directory,
    _load_json,
    _resolve_inside,
    _same_path,
)
from market_predictor.intraday.datasets.selected_session_history import (
    verify_selected_stock_sessions,
)
from market_predictor.intraday.datasets.selection import (
    INTRADAY_SELECTION_SCHEMA,
    _load_sp500_membership_eligibility,
    load_complete_intraday_selection,
)
from market_predictor.intraday.datasets.transformations import (
    _collection_artifacts,
    _load_coverage_tables,
    _membership_sector_exclusions,
    _normalize_selection,
)
from market_predictor.intraday.features.features import (
    FEATURE_SCHEMA_VERSION,
)
from market_predictor.intraday.features.labels import (
    LABEL_SCHEMA_VERSION,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)


def _verify_inputs(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    intraday_contract_lineage_path: Path = DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
) -> _VerifiedInputs:
    contract_from_disk = load_strategy_contract(strategy_contract_path)
    if (
        contract_from_disk.model_dump(mode="json") != strategy_contract.model_dump(mode="json")
        or contract_from_disk.sha256() != strategy_contract.sha256()
    ):
        raise DataReadinessError("strategy contract object differs from its frozen file")
    contract_sha256 = strategy_contract.sha256()
    selection_manifest = load_complete_intraday_selection(selection_directory)
    if selection_manifest.get("schema") != INTRADAY_SELECTION_SCHEMA:
        raise DataReadinessError("legacy or leaked intraday selection schema is prohibited")
    selection, selection_identity = verify_selected_stock_sessions(selection_directory)
    if selection_identity["strategy_id"] != strategy_contract.intraday.strategy_id:
        raise DataReadinessError("selection does not use the frozen intraday contract")
    selection_contract_identity = require_intraday_contract_lineage(
        observed_contract_sha256=selection_identity["strategy_contract_sha256"],
        observed_contract_file_sha256=None,
        current_contract=strategy_contract,
        current_contract_path=strategy_contract_path,
        lineage_path=intraday_contract_lineage_path,
    )
    selection = _normalize_selection(selection)

    stock_manifest = load_complete_intraday_history_collection(stock_collection_directory)
    stock_request = _load_json(stock_collection_directory / "_request.json")
    _require_collection_request(stock_request, timeframe="1Min", label="stock")
    coverage_manifest = load_complete_one_minute_coverage(stock_coverage_directory)
    if not bool(coverage_manifest.get("ready_for_feature_build")):
        raise DataReadinessError("stock one-minute coverage is not ready for feature build")
    coverage_contract_identity = require_intraday_contract_lineage(
        observed_contract_sha256=coverage_manifest.get(
            "strategy_contract_sha256"
        ),
        observed_contract_file_sha256=coverage_manifest.get(
            "strategy_contract_file_sha256"
        ),
        current_contract=strategy_contract,
        current_contract_path=strategy_contract_path,
        lineage_path=intraday_contract_lineage_path,
    )
    if (
        coverage_manifest.get("collection_manifest_sha256") != file_sha256(stock_collection_directory / "_manifest.json")
        or not _same_path(coverage_manifest.get("collection_path"), stock_collection_directory)
    ):
        raise DataReadinessError("stock collection and coverage lineage differ")
    five_minute_canonical_directory = _existing_directory(
        coverage_manifest.get("five_minute_canonical_path"),
        "five-minute canonical coverage parent",
    )
    _, canonical_identity = verify_canonical_five_minute_store(
        five_minute_canonical_directory
    )
    if any(
        coverage_manifest.get(key) != expected
        for key, expected in canonical_identity.items()
    ):
        raise DataReadinessError("coverage canonical five-minute parent lineage differs")
    stock_plan_directory = _existing_directory(coverage_manifest.get("plan_path"), "stock plan")
    stock_plan = load_complete_intraday_history_plan(stock_plan_directory)
    stock_plan_request = _load_json(stock_plan_directory / "_request.json")
    stock_plan_contract_identity = require_intraday_contract_lineage(
        observed_contract_sha256=stock_plan_request.get(
            "strategy_contract_sha256"
        ),
        observed_contract_file_sha256=stock_plan_request.get(
            "strategy_contract_file_sha256"
        ),
        current_contract=strategy_contract,
        current_contract_path=strategy_contract_path,
        lineage_path=intraday_contract_lineage_path,
    )
    if (
        stock_plan.get("schema") != SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA
        or stock_manifest.get("plan_fingerprint") != stock_plan.get("plan_fingerprint")
        or stock_request.get("plan_manifest_sha256") != file_sha256(stock_plan_directory / "_manifest.json")
        or coverage_manifest.get("plan_manifest_sha256") != file_sha256(stock_plan_directory / "_manifest.json")
    ):
        raise DataReadinessError("stock collection does not descend from its verified 1m plan")
    _require_selection_lineage(stock_plan.get("selection"), selection_identity, "stock plan")

    benchmark_manifest = load_complete_intraday_history_collection(benchmark_collection_directory)
    benchmark_request = _load_json(benchmark_collection_directory / "_request.json")
    _require_collection_request(benchmark_request, timeframe="1Min", label="benchmark")
    benchmark_plan_directory = _existing_directory(benchmark_request.get("plan_path"), "benchmark plan")
    benchmark_plan = load_complete_intraday_history_plan(benchmark_plan_directory)
    benchmark_plan_request = _load_json(benchmark_plan_directory / "_request.json")
    benchmark_contract_identity = require_intraday_contract_lineage(
        observed_contract_sha256=benchmark_plan_request.get(
            "strategy_contract_sha256"
        ),
        observed_contract_file_sha256=benchmark_plan_request.get(
            "strategy_contract_file_sha256"
        ),
        current_contract=strategy_contract,
        current_contract_path=strategy_contract_path,
        lineage_path=intraday_contract_lineage_path,
    )
    if (
        benchmark_plan.get("schema") != SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA
        or benchmark_manifest.get("plan_fingerprint") != benchmark_plan.get("plan_fingerprint")
        or benchmark_request.get("plan_manifest_sha256") != file_sha256(benchmark_plan_directory / "_manifest.json")
    ):
        raise DataReadinessError("benchmark collection lineage is invalid")
    _require_selection_lineage(benchmark_plan.get("selection"), selection_identity, "benchmark plan")
    benchmark_tickers = frozenset(
        str(value).upper().strip() for value in cast(list[object], benchmark_plan_request.get("benchmark_tickers", []))
    )
    if benchmark_tickers != _REQUIRED_BENCHMARKS:
        raise DataReadinessError("benchmark plan must contain SPY, QQQ, and all sector ETFs")

    market_sessions = tuple(sorted(pd.to_datetime(selection["session_date_et"], errors="raise").dt.date.unique()))
    calendar = xcals.get_calendar("XNYS")
    membership_identity = _load_sp500_membership_eligibility(
        membership_authority_directory,
        market_sessions=market_sessions,
        calendar=calendar,
    )
    if (
        selection_manifest.get("membership_authority_sha256") != membership_identity.authority_sha256
        or selection_manifest.get("membership_manifest_sha256") != membership_identity.manifest_sha256
        or selection_manifest.get("membership_table_sha256") != membership_identity.membership_table_sha256
        or selection_manifest.get("membership_universe_sha256") != membership_identity.universe_sha256
        or selection_manifest.get("membership_universe_snapshot_id") != membership_identity.universe_snapshot_id
    ):
        raise DataReadinessError("selection and PIT membership authority lineage differ")
    membership_manifest = load_plan_json(membership_authority_directory / "_manifest.json")
    membership_record = cast(Mapping[str, Any], membership_manifest["membership_artifact"])
    membership_path = _resolve_inside(membership_authority_directory, str(membership_record["path"]))
    memberships, _ = load_canonical_artifact(membership_path, expected_type="memberships", allow_research=True)

    coverage, excluded = _load_coverage_tables(stock_coverage_directory, coverage_manifest)
    incomplete_pairs = _validate_coverage(selection, coverage, excluded)
    membership_sector_excluded = _membership_sector_exclusions(
        memberships,
        selected_tickers=set(selection["ticker"].astype(str)),
    )
    all_excluded = frozenset(excluded).union(membership_sector_excluded)
    selected_security_count = int(selection["ticker"].nunique())
    if (
        selected_security_count <= 0
        or len(all_excluded) / selected_security_count
        > MAXIMUM_SECURITY_EXCLUSION_FRACTION
    ):
        raise DataReadinessError(
            "combined intraday whole-security exclusions exceed 5%"
        )
    stock_artifacts = _collection_artifacts(stock_collection_directory, stock_manifest)
    benchmark_artifacts = _collection_artifacts(benchmark_collection_directory, benchmark_manifest)
    parent_lineage = {
        "selection_authority_sha256": file_sha256(selection_directory / "_authority.json"),
        "selection_manifest_sha256": file_sha256(selection_directory / "_manifest.json"),
        "selection_table_sha256": str(selection_identity["table_sha256"]),
        "stock_collection_authority_sha256": file_sha256(stock_collection_directory / "_authority.json"),
        "stock_collection_manifest_sha256": file_sha256(stock_collection_directory / "_manifest.json"),
        "stock_coverage_authority_sha256": file_sha256(stock_coverage_directory / "_authority.json"),
        "stock_coverage_manifest_sha256": file_sha256(stock_coverage_directory / "_manifest.json"),
        "five_minute_canonical_authority_sha256": canonical_identity[
            "five_minute_canonical_authority_sha256"
        ],
        "five_minute_canonical_manifest_sha256": canonical_identity[
            "five_minute_canonical_manifest_sha256"
        ],
        "five_minute_canonical_file_inventory_sha256": canonical_identity[
            "five_minute_canonical_file_inventory_sha256"
        ],
        "benchmark_collection_authority_sha256": file_sha256(benchmark_collection_directory / "_authority.json"),
        "benchmark_collection_manifest_sha256": file_sha256(benchmark_collection_directory / "_manifest.json"),
        "membership_authority_sha256": file_sha256(membership_authority_directory / "_authority.json"),
        "membership_manifest_sha256": file_sha256(membership_authority_directory / "_manifest.json"),
        "membership_table_sha256": membership_identity.membership_table_sha256,
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": contract_sha256,
        "intraday_data_contract_sha256": (
            selection_contract_identity.intraday_data_contract_sha256
        ),
        "intraday_parent_contract_sha256": (
            selection_contract_identity.observed_contract_sha256
        ),
        "intraday_contract_lineage_file_sha256": str(
            selection_contract_identity.lineage_file_sha256 or ""
        ),
    }
    if (
        coverage_contract_identity.intraday_data_contract_sha256
        != selection_contract_identity.intraday_data_contract_sha256
        or stock_plan_contract_identity.intraday_data_contract_sha256
        != selection_contract_identity.intraday_data_contract_sha256
        or benchmark_contract_identity.intraday_data_contract_sha256
        != selection_contract_identity.intraday_data_contract_sha256
    ):
        raise DataReadinessError("intraday parent authorities use different scoped contracts")
    return _VerifiedInputs(
        selection=selection,
        coverage=coverage,
        excluded_tickers=all_excluded,
        membership_sector_excluded_tickers=membership_sector_excluded,
        incomplete_pairs=incomplete_pairs,
        memberships=memberships,
        stock_artifacts=stock_artifacts,
        benchmark_artifacts=benchmark_artifacts,
        benchmark_tickers=benchmark_tickers,
        parent_lineage=parent_lineage,
        contract=strategy_contract,
        contract_sha256=contract_sha256,
    )

def _validate_no_leakage(frame: pd.DataFrame) -> None:
    if (
        not frame["feature_schema_version"].astype(str).eq(FEATURE_SCHEMA_VERSION).all()
        or not frame["label_schema_version"].astype(str).eq(LABEL_SCHEMA_VERSION).all()
    ):
        raise DataReadinessError("dataset contains an unrecognized feature or label schema")
    eligible = frame["label_eligible"].astype(bool)
    rows = frame.loc[eligible]
    if rows.empty:
        return
    feature_at = pd.to_datetime(rows["feature_available_at_utc"], utc=True, errors="raise")
    entry_at = pd.to_datetime(rows["entry_time_utc"], utc=True, errors="raise")
    exit_end = pd.to_datetime(rows["exit_bar_end_utc"], utc=True, errors="raise")
    label_at = pd.to_datetime(rows["label_available_at_utc"], utc=True, errors="raise")
    session_close = pd.to_datetime(rows["session_close_utc"], utc=True, errors="raise")
    if (
        bool(feature_at.ge(entry_at).any())
        or bool(entry_at.ge(exit_end).any())
        or bool(exit_end.gt(label_at).any())
        or bool(exit_end.gt(session_close).any())
    ):
        raise DataReadinessError("dataset contains leakage or invalid label availability timestamps")

def _validate_monthly_partition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_stock_sessions_by_month: Mapping[str, int],
) -> None:
    if not records:
        raise DataReadinessError("intraday dataset has no monthly partitions")
    months: list[str] = []
    for record in records:
        month = str(record.get("session_month_et", ""))
        try:
            month_start = date.fromisoformat(f"{month}-01")
        except ValueError as exc:
            raise DataReadinessError(
                "intraday dataset has an invalid monthly partition identity"
            ) from exc
        if month_start.strftime("%Y-%m") != month:
            raise DataReadinessError(
                "intraday dataset has a noncanonical monthly partition identity"
            )
        expected_path = (
            f"partitions/session_month_et={month}/part-00000.parquet"
        )
        first_session = str(record.get("first_session_date_et", ""))
        last_session = str(record.get("last_session_date_et", ""))
        try:
            first = date.fromisoformat(first_session)
            last = date.fromisoformat(last_session)
        except ValueError as exc:
            raise DataReadinessError(
                f"intraday monthly partition {month} has invalid session bounds"
            ) from exc
        rows = int(record.get("rows", -1))
        eligible_rows = int(record.get("eligible_rows", -1))
        if (
            str(record.get("path", "")) != expected_path
            or first.strftime("%Y-%m") != month
            or last.strftime("%Y-%m") != month
            or first > last
            or rows < 1
            or eligible_rows < 0
            or eligible_rows > rows
            or int(record.get("stock_sessions", -1)) < 1
            or int(record.get("ticker_count", -1)) < 1
            or month not in expected_stock_sessions_by_month
            or int(record.get("stock_sessions", -1))
            > int(expected_stock_sessions_by_month.get(month, -1))
        ):
            raise DataReadinessError(
                f"intraday monthly partition {month} violates its layout contract"
            )
        months.append(month)
    if months != sorted(set(months)):
        raise DataReadinessError(
            "intraday dataset must contain at most one ordered file per month"
        )
    first_period = pd.Period(months[0], freq="M")
    last_period = pd.Period(months[-1], freq="M")
    maximum_files = int(last_period.ordinal - first_period.ordinal + 1)
    if len(records) > maximum_files:
        raise DataReadinessError(
            "intraday monthly partition count exceeds the calendar-month span"
        )

def _verify_monthly_partition_files(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    canonical_schema: pa.Schema | None = None
    previous_session: date | None = None
    required_columns = {
        "session_date_et",
        "ticker",
        "volume_bar_number",
        "dataset_eligible",
    }
    for record in records:
        month = str(record["session_month_et"])
        path = _resolve_inside(root, str(record["path"]))
        parquet = _PARQUET_FILE(path)
        schema = parquet.schema_arrow.remove_metadata()
        if canonical_schema is None:
            canonical_schema = schema
        elif not schema.equals(canonical_schema):
            raise DataReadinessError(
                "intraday monthly partition schemas differ across the publication"
            )
        if not required_columns.issubset(schema.names):
            raise DataReadinessError(
                f"intraday monthly partition omits replay columns: {path}"
            )

        file_rows = 0
        eligible_rows = 0
        stock_sessions = 0
        tickers: set[str] = set()
        first_session: date | None = None
        last_session: date | None = None
        for row_group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group_index,
                columns=[
                    "session_date_et",
                    "ticker",
                    "volume_bar_number",
                    "dataset_eligible",
                ],
            )
            if len(table) < 1:
                raise DataReadinessError(
                    f"intraday monthly partition has an empty row group: {path}"
                )
            sessions = {str(value) for value in table["session_date_et"].to_pylist()}
            if len(sessions) != 1:
                raise DataReadinessError(
                    f"intraday row group does not contain exactly one session: {path}"
                )
            session = date.fromisoformat(sessions.pop())
            if session.strftime("%Y-%m") != month:
                raise DataReadinessError(
                    f"intraday row group is stored in the wrong month: {path}"
                )
            if previous_session is not None and session <= previous_session:
                raise DataReadinessError(
                    "intraday row groups are not in strictly increasing session order"
                )
            previous_session = session
            ticker_values = [str(value) for value in table["ticker"].to_pylist()]
            bar_numbers = [int(value) for value in table["volume_bar_number"].to_pylist()]
            ordering = list(zip(ticker_values, bar_numbers, strict=True))
            if ordering != sorted(ordering):
                raise DataReadinessError(
                    f"intraday row group rows are not deterministically ordered: {path}"
                )
            group_tickers = set(ticker_values)
            tickers.update(group_tickers)
            stock_sessions += len(group_tickers)
            eligible_rows += sum(
                value is True for value in table["dataset_eligible"].to_pylist()
            )
            file_rows += len(table)
            first_session = first_session or session
            last_session = session

        if (
            parquet.metadata.num_row_groups < 1
            or file_rows != int(record["rows"])
            or eligible_rows != int(record["eligible_rows"])
            or stock_sessions != int(record["stock_sessions"])
            or len(tickers) != int(record["ticker_count"])
            or first_session is None
            or last_session is None
            or first_session.isoformat() != str(record["first_session_date_et"])
            or last_session.isoformat() != str(record["last_session_date_et"])
        ):
            raise DataReadinessError(
                f"intraday monthly partition physical counts differ: {path}"
            )

def _validate_coverage(
    selection: pd.DataFrame,
    coverage: pd.DataFrame,
    excluded: set[str],
) -> frozenset[tuple[str, str]]:
    if bool(coverage.duplicated(["ticker", "session_date_et"]).any()):
        raise DataReadinessError("coverage repeats a selected stock-session")
    selected_keys = set(zip(selection["session_date_et"], selection["ticker"], strict=True))
    coverage_keys = set(zip(coverage["session_date_et"], coverage["ticker"], strict=True))
    if selected_keys != coverage_keys:
        raise DataReadinessError("coverage does not exactly match causal selection")
    if not excluded.issubset(set(selection["ticker"])):
        raise DataReadinessError("coverage excludes a security absent from selection")
    usable = coverage[~coverage["ticker"].isin(excluded)].copy()
    observed = pd.to_numeric(usable["observed_rows"], errors="coerce")
    if bool(observed.isna().any()) or bool(observed.le(0).any()):
        raise DataReadinessError("non-excluded stock-session one-minute coverage is empty")
    status = usable["coverage_status"].astype(str)
    if bool(~status.isin({"complete", "incomplete"}).any()):
        raise DataReadinessError("stock-session coverage status is invalid")
    incomplete = usable.loc[status.eq("incomplete"), ["session_date_et", "ticker"]]
    return frozenset(
        (str(row.session_date_et), str(row.ticker))
        for row in incomplete.itertuples(index=False)
    )

def _require_selection_lineage(raw: object, expected: Mapping[str, object], label: str) -> None:
    if not isinstance(raw, Mapping):
        raise DataReadinessError(f"{label} has no selection lineage")
    for key in (
        "manifest_sha256",
        "request_sha256",
        "table_sha256",
        "strategy_id",
        "strategy_contract_sha256",
    ):
        if raw.get(key) != expected.get(key):
            raise DataReadinessError(f"{label} selection lineage differs at {key}")

def _require_collection_request(request: Mapping[str, Any], *, timeframe: str, label: str) -> None:
    if (
        request.get("provider") != "alpaca"
        or request.get("timeframe") != timeframe
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
    ):
        raise DataReadinessError(f"{label} collection must be Alpaca SIP/all {timeframe}")
