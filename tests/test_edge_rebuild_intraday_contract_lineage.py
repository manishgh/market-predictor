from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_contract_lineage import (
    intraday_data_contract_sha256,
    require_intraday_contract_lineage,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.v3.errors import DataReadinessError

CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")
LINEAGE_PATH = Path("configs/edge_rebuild_intraday_contract_lineage.toml")
PARENT_SNAPSHOT_PATH = Path(
    "configs/lineage/"
    "edge_rebuild_strategy_contract."
    "466fe894e927cfb1d3d092229c3b392c61bc41f4.toml"
)
PARENT_COMMIT = "466fe894e927cfb1d3d092229c3b392c61bc41f4"
PARENT_CONTRACT_SHA256 = (
    "8ab982460ec77c391047c4dd32c5ee3e2fea758ea448cd758cc785e3c86ce4e8"
)
PARENT_FILE_SHA256 = (
    "250a4faa7977a3d82bfacff425d56fa9193bb144be286a1c45bc2c0a65d48246"
)


def test_checked_in_parent_snapshot_mechanically_proves_lineage() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    historical = load_strategy_contract(PARENT_SNAPSHOT_PATH)

    assert file_sha256(PARENT_SNAPSHOT_PATH) == PARENT_FILE_SHA256
    assert historical.sha256() == PARENT_CONTRACT_SHA256
    assert intraday_data_contract_sha256(historical) == (
        "c88f1a2c1eb3cc3065f5a7bc38d97662da546260c760b45be321aa1718a50b39"
    ) == intraday_data_contract_sha256(contract)


def test_intraday_change_moves_scoped_identity() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    changed = contract.model_copy(
        update={
            "intraday": contract.intraday.model_copy(
                update={"round_trip_cost_bps": 11.0}
            )
        }
    )

    assert intraday_data_contract_sha256(changed) != (
        intraday_data_contract_sha256(contract)
    )


def test_explicit_parent_lineage_is_accepted() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    identity = require_intraday_contract_lineage(
        observed_contract_sha256=PARENT_CONTRACT_SHA256,
        observed_contract_file_sha256=PARENT_FILE_SHA256,
        current_contract=contract,
        current_contract_path=CONTRACT_PATH,
        lineage_path=LINEAGE_PATH,
    )

    assert identity.mode == "verified_scope_equivalent_parent"
    assert identity.source_commit == PARENT_COMMIT
    assert identity.intraday_data_contract_sha256 == (
        "c88f1a2c1eb3cc3065f5a7bc38d97662da546260c760b45be321aa1718a50b39"
    )


def test_unknown_or_wrong_file_parent_is_rejected() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    with pytest.raises(DataReadinessError, match="neither current nor explicitly"):
        require_intraday_contract_lineage(
            observed_contract_sha256="0" * 64,
            observed_contract_file_sha256=None,
            current_contract=contract,
            current_contract_path=CONTRACT_PATH,
            lineage_path=LINEAGE_PATH,
        )
    with pytest.raises(DataReadinessError, match="neither current nor explicitly"):
        require_intraday_contract_lineage(
            observed_contract_sha256=PARENT_CONTRACT_SHA256,
            observed_contract_file_sha256="0" * 64,
            current_contract=contract,
            current_contract_path=CONTRACT_PATH,
            lineage_path=LINEAGE_PATH,
        )


def test_tampered_parent_snapshot_is_rejected(tmp_path: Path) -> None:
    temporary_configs = tmp_path / "configs"
    temporary_lineage = temporary_configs / "lineage"
    temporary_lineage.mkdir(parents=True)
    lineage_path = temporary_configs / LINEAGE_PATH.name
    snapshot_path = temporary_lineage / PARENT_SNAPSHOT_PATH.name
    shutil.copyfile(LINEAGE_PATH, lineage_path)
    shutil.copyfile(PARENT_SNAPSHOT_PATH, snapshot_path)
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="snapshot file hash"):
        require_intraday_contract_lineage(
            observed_contract_sha256=PARENT_CONTRACT_SHA256,
            observed_contract_file_sha256=PARENT_FILE_SHA256,
            current_contract=load_strategy_contract(CONTRACT_PATH),
            current_contract_path=CONTRACT_PATH,
            lineage_path=lineage_path,
        )


def test_short_parent_commit_is_rejected(tmp_path: Path) -> None:
    temporary_configs = tmp_path / "configs"
    temporary_lineage = temporary_configs / "lineage"
    temporary_lineage.mkdir(parents=True)
    lineage_path = temporary_configs / LINEAGE_PATH.name
    shutil.copyfile(PARENT_SNAPSHOT_PATH, temporary_lineage / PARENT_SNAPSHOT_PATH.name)
    lineage_path.write_text(
        LINEAGE_PATH.read_text(encoding="utf-8").replace(PARENT_COMMIT, "466fe89"),
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="full immutable Git SHA"):
        require_intraday_contract_lineage(
            observed_contract_sha256=PARENT_CONTRACT_SHA256,
            observed_contract_file_sha256=PARENT_FILE_SHA256,
            current_contract=load_strategy_contract(CONTRACT_PATH),
            current_contract_path=CONTRACT_PATH,
            lineage_path=lineage_path,
        )
