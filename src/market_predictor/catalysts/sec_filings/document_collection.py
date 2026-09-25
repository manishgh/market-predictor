"""Resumable collection of EDGAR filing detail pages and the documents they name.

Phase one fetches each selected filing's EDGAR detail page and checks its header against
the pinned filing metadata; phase two fetches the primary document and every EX-99 exhibit
named by a verified page. A header phase keeps any filing's detail page whose header parses,
for checks made later (the acceptance clock). A few workers share one request governor; the
first 403 or 429 stops new requests. Bodies live in immutable zip shards beside self-hashed
receipt rows; an atomically replaced checkpoint lists shard hashes. Retrieval time is first
observation, never historical availability. EDGAR filings are immutable after acceptance.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.sources.http import HttpByteResponse
from market_predictor.sources.sec import _retry_after_seconds

SELECTED_ITEMS = frozenset({"2.02", "7.01", "8.01"})
CURRENT_REPORTS = ("8-K", "8-K/A")
MAXIMUM_BODY_BYTES = 16 * 1024 * 1024
MAXIMUM_ATTEMPTS = 3
SHARD_ATTEMPTS = 500
SHARD_BYTES = 256 * 1024 * 1024
TERMINAL_STATES = frozenset({"archived", "rejected", "missing", "oversize", "http_error"})
PHASES = ("index", "document", "header")
RECEIPT_COLUMNS = (
    "unit_id", "phase", "sec_cik", "accession_number", "sequence", "document_type", "url", "attempt", "started_at_utc",
    "completed_at_utc", "state", "reason", "status_code", "final_url", "redirect_chain_json", "retrieved_at_utc",
    "content_type", "content_encoding", "etag", "last_modified", "safe_headers_json", "body_length", "body_sha256", "member",
    "index_body_sha256", "index_row_json", "receipt_sha256",
)
UNIT_COLUMNS = ("sec_cik", "filer_ciks", "accession_number", "sec_form", "item_codes", "report_date", "filing_date",
                "accepted_at_utc", "primary_document", "url")
_METADATA = ("sec_form", "item_codes", "report_date", "filing_date", "accepted_at_utc", "primary_document")
OVERSIZE_MESSAGE = "HTTP response exceeds maximum_body_bytes"
_INTEGERS = frozenset({"attempt", "status_code", "body_length"})
_HEADER = ("Seq", "Description", "Document", "Type", "Size")
_NEW_YORK = "America/New_York"
_ITEM = re.compile(r"Item (\d{1,2}\.\d{2})")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_ARCHIVE_PATH = re.compile(r"/Archives/edgar/data/(\d{1,10})/(\d{18})/([^/]+)")

Fetch = Callable[[str], HttpByteResponse]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _pinned_json(path: Path, sha256: str) -> dict[str, object]:
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == sha256, f"SEC document file pin differs: {path.name}")
    return parse_strict_json_object(payload, label=path.name)


def archive_folder(cik: str, accession: str) -> str:
    return f"/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"


def index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov{archive_folder(cik, accession)}{accession}-index.htm"


def work_list(filings: pd.DataFrame) -> pd.DataFrame:
    """One unit per selected 8-K accession; a joint filing keeps every filer CIK and one shared metadata row."""
    current = filings.loc[filings.sec_form.isin(CURRENT_REPORTS)]
    chosen = current.loc[current.item_codes.map(lambda value: bool(SELECTED_ITEMS & set(str(value).split(","))))].copy()
    chosen["accepted_at_utc"] = pd.to_datetime(chosen.accepted_at_utc, utc=True).map(lambda value: value.isoformat())
    rows = []
    for accession, group in chosen.groupby("accession_number", sort=True):
        metadata = group.loc[:, list(_METADATA)].astype(str).drop_duplicates()
        _require(len(metadata) == 1, f"one accession carries conflicting filing metadata: {accession}")
        ciks = sorted(set(group.sec_cik.astype(str)))
        rows.append({"sec_cik": ciks[0], "filer_ciks": ",".join(ciks), "accession_number": str(accession),
                     **metadata.iloc[0].to_dict(), "url": index_url(ciks[0], str(accession))})
    return pd.DataFrame(rows, columns=UNIT_COLUMNS).astype(str)


@dataclass(frozen=True)
class IndexDocument:
    sequence: str
    description: str
    filename: str
    path: str
    document_type: str
    size_bytes: int | None

    def row(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "description": self.description, "filename": self.filename, "path": self.path,
                "document_type": self.document_type, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class FilingHeader:
    filing_date: str
    accepted_at_utc: pd.Timestamp
    period_of_report: str
    item_codes: tuple[str, ...]


@dataclass(frozen=True)
class FilingIndex(FilingHeader):
    documents: tuple[IndexDocument, ...]


def _text(node: Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def _document_path(href: str, accession: str) -> tuple[str, str]:
    """An archive link into this accession under any filer's folder (co-registrants), or an inline-viewer link to one."""
    path = href[len("/ix?doc="):] if href.startswith("/ix?doc=") else href
    match = _ARCHIVE_PATH.fullmatch(path)
    _require(match is not None and match[2] == accession.replace("-", ""),
             f"filing index links outside its accession folder: {href}")
    assert match is not None
    _require(_FILENAME.fullmatch(match[3]) is not None, f"filing index link is not a plain document name: {href}")
    return path, match[3]


def _header(soup: BeautifulSoup, accession: str) -> FilingHeader:
    number = soup.find(id="secNum")
    if not isinstance(number, Tag):
        raise DataReadinessError("filing index lacks its accession number")
    _require(_text(number).endswith(accession), "filing index is for another accession")
    fields: dict[str, str] = {}
    for head in soup.find_all("div", class_="infoHead"):
        value = head.find_next_sibling("div")
        if not isinstance(value, Tag) or "info" not in (value.get("class") or []):
            raise DataReadinessError("filing index header is malformed")
        _require(_text(head) not in fields, "filing index header repeats a field")
        fields[_text(head)] = value.get_text("\n", strip=True)
    _require({"Filing Date", "Accepted"} <= set(fields), "filing index lacks its filing date or acceptance")
    try:  # EDGAR shows acceptance on its own New York clock.
        accepted = pd.Timestamp(fields["Accepted"]).tz_localize(_NEW_YORK, ambiguous="raise", nonexistent="raise")
    except ValueError as error:
        raise DataReadinessError(f"filing index acceptance is not a New York instant: {fields['Accepted']}") from error
    return FilingHeader(fields["Filing Date"], accepted.tz_convert("UTC"), fields.get("Period of Report", ""),
                        tuple(_ITEM.findall(fields.get("Items", ""))))


def parse_filing_header(body: bytes, *, accession: str) -> FilingHeader:
    """Parse one EDGAR filing detail page's header strictly; any other shape raises."""
    return _header(BeautifulSoup(body, "html.parser"), accession)


def parse_filing_index(body: bytes, *, accession: str) -> FilingIndex:
    """Parse one EDGAR filing detail page and its document table strictly; any other shape raises."""
    soup = BeautifulSoup(body, "html.parser")
    header = _header(soup, accession)
    tables = soup.find_all("table", class_="tableFile")
    documents = [table for table in tables if table.get("summary") == "Document Format Files"]
    _require(len(documents) == 1 and all(table.get("summary") in {"Document Format Files", "Data Files"} for table in tables),
             "filing index needs exactly one document table")
    rows = documents[0].find_all("tr")
    _require(bool(rows) and tuple(_text(cell) for cell in rows[0].find_all("th")) == _HEADER,
             "filing index document table header differs")
    parsed: list[IndexDocument] = []
    for row in rows[1:]:
        cells = row.find_all("td")
        _require(len(cells) == 5, "filing index document row is malformed")
        sequence, description, _, document_type, size = (_text(cell) for cell in cells)
        link = cells[2].find("a")
        if not isinstance(link, Tag) or not isinstance(link.get("href"), str):
            raise DataReadinessError("filing index document row has no link")
        path, filename = _document_path(str(link["href"]), accession)
        if not sequence and not document_type and filename == f"{accession}.txt":
            continue  # The complete submission text file is never selected.
        _require(sequence.isdigit() and bool(document_type), "filing index document row lacks sequence or type")
        parsed.append(IndexDocument(sequence, description, filename, path, document_type,
                                    int(size) if size.isdigit() else None))
    _require(len({document.sequence for document in parsed}) == len(parsed), "filing index repeats a sequence")
    return FilingIndex(header.filing_date, header.accepted_at_utc, header.period_of_report, header.item_codes,
                       tuple(parsed))


def index_rejection(index: FilingIndex, unit: Mapping[str, str]) -> str | None:
    """Why a parsed detail page contradicts the pinned filing metadata, or None."""
    checks = (
        (index.accepted_at_utc == pd.Timestamp(unit["accepted_at_utc"]), "acceptance time differs"),
        (index.filing_date == unit["filing_date"], "filing date differs"),
        (not unit["report_date"] or index.period_of_report == unit["report_date"], "period of report differs"),
        (set(index.item_codes) == set(unit["item_codes"].split(",")), "item codes differ"),
        (sum(document.filename == unit["primary_document"] for document in index.documents) == 1,
         "primary document is not listed exactly once"),
    )
    return next((reason for passed, reason in checks if not passed), None)


def selected_documents(index: FilingIndex, primary_document: str) -> list[IndexDocument]:
    """The primary document and every EX-99 exhibit; never XBRL, graphics or the full submission."""
    return [document for document in index.documents
            if document.filename == primary_document or document.document_type.upper().startswith("EX-99")]


def _receipt(values: dict[str, Any]) -> dict[str, Any]:
    row = {column: values.get(column) for column in RECEIPT_COLUMNS[:-1]}
    row["receipt_sha256"] = json_sha256(row)
    return row


def attempt(unit: Mapping[str, Any], fetch: Fetch, *, number: int,
            now: Callable[[], datetime]) -> tuple[dict[str, Any], bytes | None]:
    """One request and its classified outcome; the caller decides whether the run continues."""
    base = {key: unit[key] for key in ("unit_id", "phase", "sec_cik", "accession_number", "sequence", "document_type",
                                       "url", "index_body_sha256", "index_row_json")}
    base.update(attempt=number, started_at_utc=now().isoformat())
    try:
        response = fetch(str(unit["url"]))
    except (RuntimeError, OSError) as error:
        message = str(error)
        state = "oversize" if message.startswith(OVERSIZE_MESSAGE) else "retryable"
        return _receipt({**base, "completed_at_utc": now().isoformat(), "state": state, "reason": message[:500]}), None
    _require(response.requested_url == unit["url"] and len(response.body) == response.body_length,
             "SEC response does not belong to its requested unit")
    status = response.status_code
    body: bytes | None = response.body
    reason = None
    if status == 200 and response.final_url == unit["url"] and not response.redirect_chain:
        state = "archived"
        if unit["phase"] in {"index", "header"}:
            accession = str(unit["accession_number"])
            try:
                if unit["phase"] == "index":
                    reason = index_rejection(parse_filing_index(response.body, accession=accession), unit["metadata"])
                else:
                    parse_filing_header(response.body, accession=accession)  # Its consumer judges the page's meaning.
            except DataReadinessError as error:
                reason = str(error)
            state = "archived" if reason is None else "rejected"
    elif status in {403, 429}:
        state, body = "stopped", None
    elif status in {404, 410}:
        state = "missing"
    elif status >= 500:
        state, body = "retryable", None
    else:
        state, reason = "http_error", f"status {status} or redirect"
    values = {**base, "completed_at_utc": now().isoformat(), "state": state, "reason": reason, "status_code": status,
              "final_url": response.final_url, "redirect_chain_json": json.dumps(list(response.redirect_chain)),
              "retrieved_at_utc": response.retrieved_at_utc.isoformat(), "content_type": response.content_type,
              "content_encoding": response.content_encoding, "etag": response.etag, "last_modified": response.last_modified,
              "safe_headers_json": json.dumps(dict(response.safe_headers), sort_keys=True)}
    if body is not None:
        values.update(body_length=len(body), body_sha256=response.sha256,
                      member=f"{unit['accession_number']}/{unit['sequence']}/{response.sha256}")
    return _receipt(values), body


class Store:
    """Immutable shards and the checkpoint that lists them."""

    def __init__(self, output: Path, shards: dict[str, dict[str, Any]], cooldown_until: str | None = None) -> None:
        self.output = output
        self.shards = shards
        self.cooldown_until = cooldown_until

    @classmethod
    def open(cls, output: Path, checkpoint_sha256: str, request_sha256: str) -> Store:
        """Reopen at the pinned checkpoint; files it does not name were never committed and are removed."""
        checkpoint = _pinned_json(output / "_checkpoint.json", checkpoint_sha256)
        shards = checkpoint.get("shards")
        if not isinstance(shards, dict) or checkpoint.get("request_sha256") != request_sha256:
            raise DataReadinessError("SEC document checkpoint belongs to another request")
        folder = output / "shards"
        for path in [*folder.glob("*.zip"), *folder.glob("*.parquet")] if folder.exists() else []:
            if path.stem not in shards:
                path.unlink()  # A crash between the shard rename and its checkpoint left an uncommitted file.
        for staged in output.glob(".*-*"):
            if staged.is_dir() and staged.name.startswith((".shard-", ".checkpoint-", ".manifest-")):
                shutil.rmtree(staged)
        for name, record in shards.items():
            for suffix in ("zip", "parquet"):
                _require(file_sha256(folder / f"{name}.{suffix}") == record[f"{suffix}_sha256"],
                         f"SEC document shard changed: {name}")
        cooldown = checkpoint.get("cooldown_until_utc")
        return cls(output, dict(shards), str(cooldown) if cooldown else None)

    def receipts(self) -> pd.DataFrame:
        frames = [pd.read_parquet(self.output / "shards" / f"{name}.parquet") for name in sorted(self.shards)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*RECEIPT_COLUMNS, "shard"])

    def flush(self, receipts: list[dict[str, Any]], bodies: dict[str, bytes], request_sha256: str) -> None:
        """Write one immutable shard and then the checkpoint that names it."""
        if not receipts:
            return
        name = f"shard-{len(self.shards):05d}"
        folder = self.output / "shards"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output, prefix=".shard-") as temporary:
            staged = Path(temporary)
            with zipfile.ZipFile(staged / f"{name}.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for member, body in sorted(bodies.items()):
                    archive.writestr(member, body)
            frame = pd.DataFrame(receipts, columns=RECEIPT_COLUMNS).assign(shard=name)
            frame.to_parquet(staged / f"{name}.parquet", index=False)
            record = {"zip_sha256": file_sha256(staged / f"{name}.zip"),
                      "parquet_sha256": file_sha256(staged / f"{name}.parquet"), "attempts": len(receipts)}
            for suffix in ("zip", "parquet"):
                os.replace(staged / f"{name}.{suffix}", folder / f"{name}.{suffix}")
        self.shards[name] = record
        self.write_checkpoint(request_sha256)

    def write_checkpoint(self, request_sha256: str) -> None:
        with tempfile.TemporaryDirectory(dir=self.output, prefix=".checkpoint-") as temporary:
            staged = Path(temporary) / "_checkpoint.json"
            write_json_object(staged, {"request_sha256": request_sha256, "shards": dict(sorted(self.shards.items())),
                                       "cooldown_until_utc": self.cooldown_until})
            os.replace(staged, self.output / "_checkpoint.json")

    def verify(self, receipts: pd.DataFrame) -> None:
        """Every receipt reproduces its hash and every stored body its recorded length and hash."""
        for row in receipts.to_dict("records"):
            digest = row.pop("receipt_sha256")
            row.pop("shard")
            clean = {key: None if value is None or (not isinstance(value, str) and pd.isna(value))
                     else int(value) if key in _INTEGERS else value for key, value in row.items()}
            _require(json_sha256(clean) == digest, f"SEC document receipt differs: {row['unit_id']}")
        for name in sorted(self.shards):
            expected = receipts.loc[receipts.shard.eq(name) & receipts.member.notna()]
            with zipfile.ZipFile(self.output / "shards" / f"{name}.zip") as archive:
                _require(sorted(archive.namelist()) == sorted(expected.member), f"SEC document shard members differ: {name}")
                for row in expected.itertuples(index=False):
                    body = archive.read(str(row.member))
                    _require(len(body) == int(row.body_length) and hashlib.sha256(body).hexdigest() == row.body_sha256,
                             f"SEC document body differs: {row.member}")


def _page_units(units: pd.DataFrame, phase: str) -> list[dict[str, Any]]:
    return [{"unit_id": f"{row['accession_number']}/{phase}", "phase": phase, "sec_cik": row["sec_cik"],
             "accession_number": row["accession_number"], "sequence": phase, "document_type": "filing_index",
             "url": row["url"], "index_body_sha256": None, "index_row_json": None, "metadata": row}
            for row in units.to_dict("records")]


def planned_units(phase: str, store: Store, receipts: pd.DataFrame, units: pd.DataFrame) -> list[dict[str, Any]]:
    _require(phase in PHASES, f"unknown SEC document phase: {phase}")
    return document_units(store, receipts, units) if phase == "document" else _page_units(units, phase)


def document_units(store: Store, receipts: pd.DataFrame, units: pd.DataFrame) -> list[dict[str, Any]]:
    """Every selected document named by an archived, verified detail page."""
    metadata = units.set_index("accession_number").to_dict("index")
    archived = receipts.loc[receipts.phase.eq("index") & receipts.state.eq("archived")].sort_values("accession_number")
    planned: list[dict[str, Any]] = []
    for name, rows in archived.groupby("shard", sort=True):
        with zipfile.ZipFile(store.output / "shards" / f"{name}.zip") as archive:
            for row in rows.to_dict("records"):
                accession = str(row["accession_number"])
                unit = {**metadata[accession], "accession_number": accession}
                index = parse_filing_index(archive.read(str(row["member"])), accession=accession)
                for document in selected_documents(index, unit["primary_document"]):
                    planned.append({
                        "unit_id": f"{accession}/{document.sequence}", "phase": "document", "sec_cik": unit["sec_cik"],
                        "accession_number": accession, "sequence": document.sequence,
                        "document_type": document.document_type, "url": f"https://www.sec.gov{document.path}",
                        "index_body_sha256": row["body_sha256"],
                        "index_row_json": json.dumps(document.row(), sort_keys=True), "metadata": unit})
    return sorted(planned, key=lambda unit: str(unit["unit_id"]))


def _outstanding(units: list[dict[str, Any]], receipts: pd.DataFrame) -> list[tuple[dict[str, Any], int]]:
    terminal = set(receipts.loc[receipts.state.isin(TERMINAL_STATES), "unit_id"])
    retried = receipts.loc[receipts.state.eq("retryable")].groupby("unit_id").size().to_dict()
    tried = receipts.groupby("unit_id").size().to_dict()
    return [(unit, int(tried.get(unit["unit_id"], 0)) + 1) for unit in units
            if unit["unit_id"] not in terminal and int(retried.get(unit["unit_id"], 0)) < MAXIMUM_ATTEMPTS]


def _stop_until(stopped: list[dict[str, Any]], cooldowns: Mapping[int, float]) -> str:
    """The latest end of SEC's cooldown over every stopped attempt (its Retry-After, else the configured cooldown)."""
    ends = []
    for receipt in stopped:
        headers = json.loads(str(receipt["safe_headers_json"]))
        wait_seconds = _retry_after_seconds(headers.get("retry-after"), fallback=cooldowns[int(receipt["status_code"])])
        ends.append(datetime.fromisoformat(str(receipt["completed_at_utc"])) + timedelta(seconds=wait_seconds))
    return max(ends).isoformat()


class RequestsStopped(Exception):
    """Raised instead of sending a request once the run has stopped."""


@dataclass
class _Pass:
    receipts: list[dict[str, Any]]
    bodies: dict[str, bytes]
    stopped: list[dict[str, Any]]
    buffered: int = 0
    failure: Exception | None = None


def _run_pass(pool: ThreadPoolExecutor, pending: list[tuple[dict[str, Any], int]], *, store: Store, fetch: Fetch,
              request_sha256: str, memory_check: Callable[[], None], stop: threading.Event, workers: int,
              now: Callable[[], datetime]) -> _Pass:
    """One pass over the outstanding units, flushing full shards. The first stopped attempt or failure sets
    `stop`: nothing new is submitted, requests not yet sent are cancelled, and every finished attempt is kept."""
    queue = iter(pending)
    running: set[Future[tuple[dict[str, Any], bytes | None]]] = set()
    batch = _Pass([], {}, [])
    while True:
        while not stop.is_set() and len(running) < workers and (item := next(queue, None)) is not None:
            running.add(pool.submit(attempt, item[0], fetch, number=item[1], now=now))
        if not running:
            break
        done, running = wait(running, return_when=FIRST_COMPLETED)
        for future in done:
            try:
                receipt, body = future.result()
            except RequestsStopped:
                continue  # Never sent: the unit stays outstanding for a resume.
            except Exception as error:
                batch.failure = batch.failure or error
                stop.set()
                continue
            batch.receipts.append(receipt)
            if body is not None:
                batch.bodies[str(receipt["member"])] = body
                batch.buffered += len(body)
            if receipt["state"] == "stopped":
                batch.stopped.append(receipt)
                stop.set()
        if not stop.is_set() and (len(batch.receipts) >= SHARD_ATTEMPTS or batch.buffered >= SHARD_BYTES):
            store.flush(batch.receipts, batch.bodies, request_sha256)
            batch = _Pass([], {}, [])
            try:
                memory_check()
            except Exception as error:  # The shard above is kept; drain in-flight attempts, then re-raise.
                batch.failure = error
                stop.set()
    return batch


def collect(*, store: Store, units: pd.DataFrame, fetch: Fetch, request_sha256: str, memory_check: Callable[[], None],
            cooldowns: Mapping[int, float], stop: threading.Event, phases: Sequence[str] = ("index", "document"),
            workers: int = 1, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> dict[str, Any]:
    """Advance each phase with `workers` concurrent requests, retrying transient failures.

    The first 403 or 429 sets `stop`: attempts already sent finish and are kept, and `fetch` must
    then refuse to send (raising `RequestsStopped`), including from a governor wait. The
    checkpoint records when SEC's cooldown ends; no request is sent before then, even from a new
    process. `fetch` must be safe to call from several threads.
    """
    _require(workers >= 1, "SEC document collection needs at least one worker")
    if store.cooldown_until is not None and now() < datetime.fromisoformat(store.cooldown_until):
        raise DataReadinessError(f"SEC cooldown has not elapsed; resume after {store.cooldown_until}")
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sec-documents") as pool:
            for phase in phases:
                planned = planned_units(phase, store, store.receipts(), units)
                for _ in range(MAXIMUM_ATTEMPTS):
                    pending = _outstanding(planned, store.receipts())
                    if not pending:
                        break
                    batch = _run_pass(pool, pending, store=store, fetch=fetch, request_sha256=request_sha256,
                                      memory_check=memory_check, stop=stop, workers=workers, now=now)
                    if batch.stopped:
                        store.cooldown_until = _stop_until(batch.stopped, cooldowns)
                    store.flush(batch.receipts, batch.bodies, request_sha256)
                    if batch.failure is not None:
                        raise batch.failure
                    if batch.stopped:
                        first = min(batch.stopped, key=lambda receipt: str(receipt["completed_at_utc"]))
                        return {"status": "stopped", "stop_status_code": first["status_code"],
                                "stopped_unit": first["unit_id"]}
    finally:
        stop.set()  # Wake any governor wait so the pool can shut down.
    return {"status": "complete"}


def outcomes(store: Store, units: pd.DataFrame, phases: Sequence[str] = ("index", "document")
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each planned unit's final state (retry exhaustion included) and every receipt."""
    receipts = store.receipts()
    planned = [unit for phase in phases for unit in planned_units(phase, store, receipts, units)]
    terminal = receipts.loc[receipts.state.isin(TERMINAL_STATES)]
    _require(not terminal.unit_id.duplicated().any(), "an SEC document unit has more than one terminal attempt")
    final = dict(zip(terminal.unit_id, terminal.state, strict=True))
    retried = receipts.loc[receipts.state.eq("retryable")].groupby("unit_id").size().to_dict()
    rows = [{"unit_id": unit["unit_id"], "phase": unit["phase"], "accession_number": unit["accession_number"],
             "document_type": unit["document_type"],
             "state": final.get(unit["unit_id"], "retry_exhausted" if int(retried.get(unit["unit_id"], 0)) >= MAXIMUM_ATTEMPTS
                                else "outstanding")} for unit in planned]
    return pd.DataFrame(rows, columns=["unit_id", "phase", "accession_number", "document_type", "state"]), receipts
