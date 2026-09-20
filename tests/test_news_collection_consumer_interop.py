"""Opt-in offline interoperability against a locally built TradingFlow CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from market_predictor.evidence.news_collection import NewsCollectionPlan, collect_news_receipts
from market_predictor.evidence.news_exchange import validate_news_receipt
from market_predictor.locking import file_lock


@pytest.mark.skipif(os.name != "nt", reason="current shared inbox adapter uses Windows local-disk locking")
def test_python_publication_csharp_import_lock_and_restart(tmp_path: Path) -> None:
    configured = os.environ.get("TRADINGFLOW_NEWS_CLI_DLL")
    if configured is None:
        pytest.skip("set TRADINGFLOW_NEWS_CLI_DLL to the independently built TradingFlow CLI")
    dll = Path(configured)
    assert dll.is_absolute() and dll.is_file(), "configured TradingFlow CLI must exist"
    vector = json.loads((Path(__file__).parent / "fixtures/news_collection_exchange.json").read_text(encoding="utf-8"))
    plan = NewsCollectionPlan.model_validate_json(vector["plan_utf8"])
    receipt = validate_news_receipt(vector["manifest_utf8"].encode(), vector["payload_utf8"].encode())
    source, inbox = tmp_path / "collector", tmp_path / "inbox"
    report = collect_news_receipts(root=source, plan=plan, expected_plan_sha256=vector["plan_sha256"],
        fetch_page=lambda request: receipt)
    assert report.complete
    command = ["dotnet", str(dll), "import-shared-news", "--trusted-collector-root", str(source),
        "--plan-sha256", vector["plan_sha256"], "--inbox", str(inbox)]

    def invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW)

    with file_lock(source / "collection-owner", timeout=0):
        busy = invoke()
        assert busy.returncode != 0, busy.stdout
        assert not inbox.exists(), "contention must not acknowledge or create an inbox"
    # Abrupt owner death must release the OS lock without deleting/replacing it.
    owner = subprocess.Popen([sys.executable, "-u", "-c",
        "import os,sys,msvcrt,time; f=os.open(sys.argv[1],os.O_RDWR); "
        "msvcrt.locking(f,msvcrt.LK_NBLCK,1); print('locked',flush=True); time.sleep(20)",
        str(source / "collection-owner.lock")], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        assert owner.stdout is not None and owner.stdout.readline().strip() == "locked"
        assert invoke().returncode != 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)
    first = invoke()
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["Imported"] == 1
    second = invoke()
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["AlreadyImported"] == 1
    assert not json.loads(second.stdout)["tradingAdmitted"]
    assert collect_news_receipts(root=source, plan=plan, expected_plan_sha256=vector["plan_sha256"]) == report
    imported = list(inbox.glob("runs/*/receipts/*/payload.json"))
    assert len(imported) == 1 and imported[0].read_bytes() == receipt.payload_bytes
    imported[0].write_bytes(b"{}")
    corrupt = invoke()
    assert corrupt.returncode != 0
    assert imported[0].read_bytes() == b"{}", "corruption must not be overwritten as successful recovery"
