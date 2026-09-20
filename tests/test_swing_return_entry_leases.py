"""All research entry points share the configured lease before loading inputs."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import swing_return_training as training
from market_predictor.research import swing_training_readiness as readiness


@pytest.mark.parametrize("entry", ["readiness", "training"])
@pytest.mark.parametrize("absolute", [False, True])
def test_configured_lease_blocks_before_input_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, absolute: bool,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    elsewhere = tmp_path / "different-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runtime = root / "custom-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime) if absolute else "custom-runtime")
    module = readiness if entry == "readiness" else training
    command = readiness.audit_swing_training_readiness if entry == "readiness" else training.train_swing_returns
    read = Mock(side_effect=AssertionError("busy lease must reject before input reads"))
    monkeypatch.setattr(module, "pinned_object", read)
    output = root / ("data/reports/readiness.json" if entry == "readiness" else "data/research/run")
    with heavy_job_lease("existing-job", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            command(root=root, config=Path("missing-config.json"), config_sha256="a" * 64, output=output)
    read.assert_not_called()
    assert not (elsewhere / "custom-runtime").exists()
    assert not (root / "data/runtime").exists()
