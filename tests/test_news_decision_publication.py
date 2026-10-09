"""Synthetic UNIT publication fixtures only; no real-source or admission claims."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.research import news_decision_features as f
from market_predictor.research import news_decision_publication as n
from market_predictor.research.news_source_links import SourceLinkedNewsSlice, SourceLinkUncertainty
from market_predictor.swing.features.research_partition import ResearchFeaturePartition

CUT = datetime(2024, 1, 10, 22, tzinfo=UTC)
MODEL = (*f.TECHNICAL_COLUMNS, *(f"parent_feature_{i}" for i in range(121)))
CLOCKS = {name: f"available_at_{name}" for name in MODEL}


def _parent(*, cut: datetime = CUT, ids: tuple[str, ...] = ("d1", "d2")) -> ResearchFeaturePartition:
    rows = len(ids)
    frame = pd.DataFrame({
        "decision_id": pd.Series(ids, dtype="string"), "security_id": pd.Series(["A"] * rows, dtype="string"),
        "decision_time_utc": pd.to_datetime([cut + timedelta(days=i) for i in range(rows)], utc=True),
        "feature_profile": pd.Series(["original_profile"] * rows, dtype="string"),
        "target": pd.Series([np.nan if i else 0.1 for i in range(rows)], dtype="float64"),
        "sample_weight": pd.Series([0.75] * rows, dtype="float32"),
        "feature_eligible": pd.Series([i == 0 for i in range(rows)], dtype="bool"),
    })
    frame["opaque_parent_reason"] = pd.Series([None if i else "retained" for i in range(rows)], dtype="string")
    values = {name: pd.Series([float(i + 1)] * rows, dtype="float32") for i, name in enumerate(MODEL)}
    clocks = {name: pd.to_datetime([cut - timedelta(hours=1)] * rows, utc=True) for name in CLOCKS.values()}
    frame = pd.concat([frame, pd.DataFrame(values), pd.DataFrame(clocks)], axis=1)
    return ResearchFeaturePartition(frame, MODEL, CLOCKS, {})


def _source(*, known: datetime = CUT - timedelta(hours=2)) -> SourceLinkedNewsSlice:
    version = f.NewsVersion("version", "alpaca", "document", "a" * 64, "b" * 64,
                            known - timedelta(hours=1), known, known, True,
                            ("earnings", "guidance"),
                            (f.DecisionCue("earnings", "up", "announced"), f.DecisionCue("guidance", "down", "announced")),
                            False, 0.8, known, "historical_proxy")
    link = f.VerifiedCompanyLink("version", "A", known, known, "c" * 64, "historical_proxy")
    return SourceLinkedNewsSlice((version,), (link,), ())


class Reader:
    def __init__(self, source: SourceLinkedNewsSlice) -> None:
        self.source = source
        self.calls: list[tuple[str, datetime, datetime]] = []

    def for_period(self, security_id: str, start_utc: datetime, end_utc: datetime, *, max_versions: int = 100000) -> SourceLinkedNewsSlice:
        self.calls.append((security_id, start_utc, end_utc))
        return self.source


@pytest.fixture(autouse=True)
def _unit_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(n.parent_inputs, "_guard", lambda: None)


@pytest.mark.parametrize("infer_string", [False, True])
def test_exact_parent_and_kernel_values_clocks_dtypes_and_single_parity(infer_string: bool) -> None:
    with pd.option_context("future.infer_string", infer_string):
        parent = _parent()
        reader = Reader(_source())
        result = n.build_news_month(parent=parent, reader=reader)
        assert result.model_columns == (*MODEL, *f.NEWS_COLUMNS)
        n._parity(parent.rows, result.rows)
        assert all(str(result.rows[column].dtype) == "float32" for column in f.NEWS_COLUMNS)
        assert result.rows.news_earnings_up_times_guidance_down_3d.tolist() == [1.0, 1.0]
        assert result.rows.news_direct_earnings_present_3d.tolist() == [1.0, 1.0]
        assert result.rows[n.NEWS_CLOCKS["news_direct_earnings_present_3d"]].eq(CUT - timedelta(hours=2)).all()
        assert result.rows.news_direct_source_coverage_known_alpaca_3d.eq(0).all()
        singles = [n.build_news_month(parent=replace(parent, rows=parent.rows.iloc[[i]]), reader=reader).rows for i in range(2)]
        pd.testing.assert_frame_equal(result.rows, pd.concat(singles), check_exact=True)
        assert not any(result.audit[key] for key in n.CLOSED)


def test_missing_sources_remain_null_with_explicit_reason_and_known_zero_indicators() -> None:
    result = n.build_news_month(parent=_parent(), reader=Reader(SourceLinkedNewsSlice((), (), ())))
    frame = result.rows
    column = "news_direct_earnings_present_3d"
    assert frame[column].isna().all()
    assert frame[n.NEWS_CLOCKS[column]].isna().all()
    assert frame[n.NEWS_REASONS[column]].eq("no_readable_window_or_verified_empty_evidence").all()
    assert frame.news_direct_source_coverage_known_alpaca_3d.eq(0).all()


@pytest.mark.parametrize("known", [None, CUT + timedelta(hours=1)])
def test_uncertainty_applies_only_after_causal_association_and_known_time(known: datetime | None) -> None:
    source = _source()
    uncertainty = SourceLinkUncertainty("copy", "alpaca", "document", known, CUT - timedelta(hours=1), "unknown_revision")
    result = n.build_news_month(parent=_parent(), reader=Reader(replace(source, uncertainties=(uncertainty,)))).rows
    if known is not None:
        assert result.loc[0, "news_direct_earnings_present_3d"] == 1
    else:
        assert result.loc[0, list(f.NEWS_COLUMNS)].isna().all()
    assert result.loc[1, list(f.NEWS_COLUMNS)].isna().all()
    assert result.loc[1, "news_projection_status"] == "source_history_unresolved"
    assert result.loc[1, "news_uncertainty_count"] == 1


def test_wrong_stock_source_slice_is_rejected() -> None:
    source = _source()
    with pytest.raises(DataReadinessError, match="another stock"):
        n.build_news_month(parent=_parent(), reader=Reader(replace(source, links=(replace(source.links[0], security_id="B"),))))


def test_future_technical_clock_is_rejected_by_shared_kernel() -> None:
    parent = _parent()
    parent.rows.loc[0, CLOCKS[f.TECHNICAL_COLUMNS[0]]] = CUT + timedelta(seconds=1)
    with pytest.raises(ValueError, match="technical"):
        n.build_news_month(parent=parent, reader=Reader(_source()))


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    months = ("2024-01", "2024-02")
    parts = {months[0]: _parent(), months[1]: _parent(cut=datetime(2024, 2, 10, 22, tzinfo=UTC), ids=("d3", "d4"))}
    source = root / "source.json"
    source.write_text("UNIT retained input", encoding="utf-8")
    kernel = root / "src/market_predictor/research/news_decision_features.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("# UNIT implementation pin", encoding="utf-8")
    implementations = {kernel.relative_to(root).as_posix(): file_sha256(kernel)}
    files = {source.name: file_sha256(source)}
    pin = n.SourcePin(path=source.name, sha256=file_sha256(source))
    config = root / "config.json"
    write_json_object(config, {"schema": n.CONFIG_SCHEMA, "source_link_index": pin.model_dump(mode="json")})
    config_pin = n.SourcePin(path=config.name, sha256=file_sha256(config))
    reads: list[str] = []

    def read_month(month: str) -> pd.DataFrame:
        reads.append(month)
        return parts[month].rows.copy()

    parent = SimpleNamespace(
        model_columns=MODEL, availability_columns=CLOCKS, source_files=files, implementation_files=implementations,
        historical_implementation_files={"original_producer": {"archived.py": "d" * 64}},
        request={"profile_sha256": "e" * 64, "cohort_sha256": "f" * 64},
        manifest={"months": {month: {"rows": 2, "decision_ids_sha256": json_sha256(sorted(part.rows.decision_id))}
                             for month, part in parts.items()}}, read_month=read_month,
        recheck=lambda: n.check_files(root, files),
    )
    reader = Reader(_source())
    reader.source_files = files  # type: ignore[attr-defined]
    reader.implementation_files = implementations  # type: ignore[attr-defined]

    @contextmanager
    def parent_context(**kwargs: Any):
        n.check_files(root, files)
        yield parent
        n.check_files(root, files)

    @contextmanager
    def reader_context(**kwargs: Any):
        yield reader
        n.check_files(root, files)

    monkeypatch.setattr(n.parent_inputs, "verified_news_parent_inputs", parent_context)
    monkeypatch.setattr(n, "open_news_source_link_index", reader_context)
    monkeypatch.setattr(n, "heavy_job_lease", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(n, "_implementation", lambda root: implementations)
    monkeypatch.setattr(n.parent_inputs, "MONTHS", months)
    monkeypatch.setattr(n.parent_inputs, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(n.parent_inputs, "EXPECTED_IDS", json_sha256(["d1", "d2", "d3", "d4"]))
    return SimpleNamespace(config=config_pin, output=Path("published"), source=source, kernel=kernel, parts=parts, reads=reads)


def test_publish_partial_then_resume_exact_parent_209_and_immutable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _fixture(tmp_path, monkeypatch)
    first = n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output, maximum_months_this_run=1)
    assert first["status"] == "partial_research_only" and first["rows"] == 2
    assert case.reads == ["2024-01"]
    with pytest.raises(DataReadinessError, match="external checkpoint"):
        n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output)
    final = n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output,
                                       expected_checkpoint_sha256=first["checkpoint_sha256"])
    assert final["status"] == "complete_research_only" and final["rows"] == 4
    assert len(final["model_columns"]) == 209
    assert all(final[key] is False for key in n.CLOSED)
    assert final["manifest_sha256"] == file_sha256(tmp_path / case.output / "_manifest.json")
    for month, record in final["months"].items():
        rows, sidecar = load_canonical_artifact(tmp_path / case.output / record["path"], expected_type=n.ARTIFACT_TYPE, allow_research=True)
        n._parity(case.parts[month].rows, rows)
        assert sidecar["artifact_path"] == record["path"]
    with pytest.raises(DataReadinessError, match="already exists"):
        n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output)


@pytest.mark.parametrize("poison", ["checkpoint", "child", "code", "config", "source", "extra"])
def test_resume_rejects_changed_inputs_or_partial_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    case = _fixture(tmp_path, monkeypatch)
    first = n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output, maximum_months_this_run=1)
    stage = tmp_path / ".published.building"
    target = {"checkpoint": stage / "_checkpoint.json", "child": stage / "2024-01" / f"{n.PROFILE}.parquet",
              "code": case.kernel, "config": tmp_path / case.config.path, "source": case.source, "extra": stage / "unknown.txt"}[poison]
    with target.open("ab") as stream:
        stream.write(b"poison")
    with pytest.raises(DataReadinessError):
        n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output,
                                   expected_checkpoint_sha256=first["checkpoint_sha256"])
    assert not (tmp_path / case.output).exists()


def test_final_source_recheck_prevents_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _fixture(tmp_path, monkeypatch)

    def mutate(update: dict[str, Any]) -> None:
        case.source.write_text("changed during UNIT run", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="source changed"):
        n.materialize_news_decisions(root=tmp_path, config=case.config, output=case.output, progress=mutate)
    assert not (tmp_path / case.output).exists()
