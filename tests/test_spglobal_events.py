from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import market_predictor.commands.v3_readiness as command_module
import market_predictor.v3.spglobal_events as event_module
from market_predictor.research_cli import app as research_app
from market_predictor.core.errors import DataReadinessError
from market_predictor.v3.spglobal_archive import VerifiedSpGlobalRawArchive


def test_event_extraction_reconciles_duplicate_source_assertions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _fake_archive(tmp_path, count=3)
    bodies = {
        archive.releases[0]["url"]: _modern_html("January 5, 2026"),
        archive.releases[1]["url"]: _modern_html("January 5, 2026"),
        archive.releases[2]["url"]: _modern_html("TBA"),
    }
    monkeypatch.setattr(
        event_module,
        "require_spglobal_raw_archive_complete",
        lambda _: archive,
    )
    monkeypatch.setattr(
        event_module,
        "read_verified_spglobal_release_html",
        lambda _, record: bodies[str(record["url"])],
    )
    output = tmp_path / "events"

    manifest = event_module.extract_spglobal_events(
        archive_directory=archive.root,
        output_directory=output,
    )

    assert manifest["status"] == "complete"
    assert manifest["parsed_release_count"] == 2
    assert manifest["no_effective_event_release_count"] == 1
    assert manifest["assertion_count"] == 4
    assert manifest["event_count"] == 2
    assert manifest["duplicate_support_count"] == 2
    events = _json(output / "events.json")
    assert {event["ticker"] for event in events} == {"AIV", "TSLA"}
    assert all(event["support_count"] == 2 for event in events)
    verified = event_module.require_spglobal_event_reconstruction_ready(
        output,
        archive_directory=archive.root,
    )
    tesla = next(change for change in verified.changes if change.ticker == "TSLA")
    assert len(tesla.source_evidence()) == 2
    assert {source.source_url for source in tesla.source_evidence()} == {
        archive.releases[0]["url"],
        archive.releases[1]["url"],
    }

    events[0]["ticker"] = "FAKE"
    event_module._atomic_json(output / "events.json", events)
    published_manifest = _json_object(output / "_manifest.json")
    published_manifest["artifacts"]["events"] = event_module._artifact_record(
        output / "events.json"
    )
    event_module._atomic_json(output / "_manifest.json", published_manifest)
    published_authority = _json_object(output / "_authority.json")
    published_authority["artifact_sha256"] = event_module._file_sha256(
        output / "_manifest.json"
    )
    published_authority["event_set_sha256"] = event_module._json_sha256(events)
    event_module._atomic_json(output / "_authority.json", published_authority)
    with pytest.raises(DataReadinessError, match="counts or identity"):
        event_module.require_spglobal_event_reconstruction_ready(
            output,
            archive_directory=archive.root,
        )


def test_event_extraction_blocks_opposite_actions_at_same_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _fake_archive(tmp_path, count=1)
    body = _modern_html("January 5, 2026").replace(
        "<td>AIV</td>",
        "<td>TSLA</td>",
    )
    monkeypatch.setattr(
        event_module,
        "require_spglobal_raw_archive_complete",
        lambda _: archive,
    )
    monkeypatch.setattr(
        event_module,
        "read_verified_spglobal_release_html",
        lambda *_: body,
    )

    manifest = event_module.extract_spglobal_events(
        archive_directory=archive.root,
        output_directory=tmp_path / "events",
    )

    assert manifest["status"] == "blocked"
    assert manifest["conflict_count"] == 1
    assert not (tmp_path / "events" / "_authority.json").exists()


def test_event_extraction_blocks_effective_date_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _fake_archive(tmp_path, count=1)
    monkeypatch.setattr(
        event_module,
        "require_spglobal_raw_archive_complete",
        lambda _: archive,
    )
    monkeypatch.setattr(
        event_module,
        "read_verified_spglobal_release_html",
        lambda *_: _modern_html("December 31, 2025"),
    )

    manifest = event_module.extract_spglobal_events(
        archive_directory=archive.root,
        output_directory=tmp_path / "events",
    )

    assert manifest["status"] == "blocked"
    assert manifest["parsed_release_count"] == 0
    assert manifest["unresolved_release_count"] == 1
    assert "precedes its official publication date" in str(
        manifest["unresolved_releases"]
    )
    assert not (tmp_path / "events" / "_authority.json").exists()


def test_event_extraction_rejects_output_inside_raw_archive(tmp_path: Path) -> None:
    archive = _fake_archive(tmp_path, count=1)
    output = archive.root / "events"

    with pytest.raises(DataReadinessError, match="must be disjoint"):
        event_module.extract_spglobal_events(
            archive_directory=archive.root,
            output_directory=output,
        )

    assert not output.exists()


def test_memory_failure_does_not_publish_event_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _fake_archive(tmp_path, count=1)
    monkeypatch.setattr(
        event_module,
        "require_spglobal_raw_archive_complete",
        lambda _: archive,
    )
    monkeypatch.setattr(
        event_module,
        "read_verified_spglobal_release_html",
        lambda *_: _modern_html("January 5, 2026"),
    )
    monkeypatch.setattr(
        event_module,
        "assert_peak_memory_budget",
        lambda **_: (_ for _ in ()).throw(DataReadinessError("memory budget")),
    )
    output = tmp_path / "events"

    with pytest.raises(DataReadinessError, match="memory budget"):
        event_module.extract_spglobal_events(
            archive_directory=archive.root,
            output_directory=output,
        )

    assert not (output / "_authority.json").exists()


def test_event_reconciliation_blocks_non_alternating_actions() -> None:
    assertions = [
        _assertion("2025-01-02T05:00:00+00:00", "addition", "TEST"),
        _assertion("2025-02-03T05:00:00+00:00", "addition", "TEST"),
    ]

    _, conflicts = event_module._canonical_events(assertions)

    assert conflicts == [
        {
            "type": "non_alternating_membership_actions",
            "ticker": "TEST",
            "action": "addition",
            "previous_effective_at_utc": "2025-01-02T05:00:00+00:00",
            "effective_at_utc": "2025-02-03T05:00:00+00:00",
        }
    ]


def test_event_cli_returns_nonzero_for_blocked_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        command_module,
        "extract_spglobal_events",
        lambda **_: {
            "status": "blocked",
            "release_count": 1,
            "parsed_release_count": 0,
            "no_effective_event_release_count": 0,
            "unresolved_release_count": 1,
            "assertion_count": 0,
            "event_count": 0,
            "duplicate_support_count": 0,
            "conflict_count": 0,
        },
    )

    result = CliRunner().invoke(
        research_app,
        [
            "extract-sp500-official-events",
            "--archive-dir",
            str(tmp_path / "raw"),
            "--out-dir",
            str(tmp_path / "events"),
        ],
    )

    assert result.exit_code == 2


def _fake_archive(tmp_path: Path, *, count: int) -> VerifiedSpGlobalRawArchive:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "_authority.json").write_text("{}\n", encoding="utf-8")
    releases = tuple(
        {
            "url": f"https://press.spglobal.com/2026-01-0{number + 1}-release",
            "published_date": f"2026-01-0{number + 1}",
            "sha256": hashlib.sha256(f"release-{number}".encode()).hexdigest(),
            "unit_id": f"release-{number}",
        }
        for number in range(count)
    )
    return VerifiedSpGlobalRawArchive(
        root=root,
        authority={"artifact_sha256": "a" * 64},
        manifest={"release_set_sha256": "b" * 64},
        releases=releases,
    )


def _modern_html(effective_date: str) -> str:
    return f"""
    <html><body><table>
      <tr>
        <th>Effective Date</th><th>Index Name</th><th>Action</th>
        <th>Company Name</th><th>Ticker</th><th>GICS Sector</th>
      </tr>
      <tr>
        <td>{effective_date}</td><td>S&amp;P 500</td><td>Addition</td>
        <td>Tesla</td><td>TSLA</td><td>Consumer Discretionary</td>
      </tr>
      <tr>
        <td></td><td>S&amp;P 500</td><td>Deletion</td>
        <td>Apartment Investment</td><td>AIV</td><td>Real Estate</td>
      </tr>
    </table></body></html>
    """


def _assertion(effective_at: str, action: str, ticker: str) -> dict[str, object]:
    return {
        "effective_at_utc": effective_at,
        "action": action,
        "ticker": ticker,
        "company": "Test Company",
        "sector": "Industrials",
        "source_url": f"https://press.spglobal.com/{effective_at[:10]}-test",
        "source_published_date": effective_at[:10],
        "source_sha256": "a" * 64,
    }


def _json(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    return value


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
