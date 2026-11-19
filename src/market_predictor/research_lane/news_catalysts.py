"""Point-in-time reader and dense session matrices for Tier-1 issuer news and SEC 8-K catalysts.

Reads only retained direct-issuer classified Alpaca news events and retained initial-fit SEC
8-K earnings releases whose availability timestamps precede the end of ``development_end``
(New York calendar day). Rows in the sealed period (2024-05-29 onward) are filtered out in
PyArrow before entering Python memory and re-verified immediately.

Every event is mapped to ``reaction_session = e``, the first XNYS session whose 16:00 ET
regular close is at or after the event's availability time. A decision made after session
``t >= e`` closes enters at ``open[t + 1]``, so neither pre-market nor after-hours news can
ever enter at a prior close.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.research_lane.panel import (
    PANEL_COLUMNS,
    PROFILE_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
    file_sha256,
)
from market_predictor.research_lane.sec_filings import (
    FILINGS_DIRECTORY,
    FILINGS_FILE,
    FILINGS_MANIFEST,
     SEALED_DIRECTORY_PREFIX,
    acceptance_cutoff,
    earnings_releases,
    event_rows,
    read_sec_filings,
)

EARLY_NEWS_FAMILY_DIR: Final = Path("data/research/issuer_event_family_20190709_20210708_v2")
LATER_NEWS_FAMILY_DIR: Final = Path("data/research/issuer_event_family_20210709_20260708_v2")
FAMILY_EVENTS_FILE: Final = "family_events.parquet"
FAMILY_MANIFEST_FILE: Final = "_manifest.json"
CATALYST_EXPIRY_SESSIONS: Final = 21

FAMILY_COLUMNS: Final = (
    "security_id",
    "ticker",
    "source_family",
    "event_family",
    "matched_text",
    "published_at_utc",
    "event_available_at_utc",
    "feature_available_at_utc",
    "relation_channel",
    "research_eligible",
)

# Priority order when multiple Tier-1 positive/structural events land on the same session:
# lower positive code = higher structural priority (1 = earnings, ..., 6 = analyst_revision).
CATEGORY_CODES: Final[dict[str, int]] = {
    "earnings": 1,
    "guidance": 2,
    "merger_acquisition": 3,
    "regulatory_decision": 4,
    "product_event": 5,
    "analyst_revision": 6,
}
CODE_TO_CATEGORY: Final[dict[int, str]] = {code: name for name, code in CATEGORY_CODES.items()}

_RETROSPECTIVE_NOISE: Final = re.compile(
    r"\b(?:why\s+.{0,60}\bshares\b|biggest\s+(?:stock\s+)?movers|stocks?\s+to\s+watch|"
    r"trending\s+stocks|market\s+roundup|most\s+active\s+stocks|market\s+clubhouse)\b",
    re.IGNORECASE,
)
_BEARISH_TEXT: Final = re.compile(
    r"\b(?:downgrades?|downgraded|lowers?|lowered|cuts?|cut|reduces?|reduced|"
    r"withdraws?|withdrawn|suspends?|suspended|rejects?|rejected|declines?|declined|"
    r"complete\s+response\s+letter|blocks?|blocked|challenges?|challenged)\b",
    re.IGNORECASE,
)

BoolMatrix = npt.NDArray[np.bool_]
Int8Matrix = npt.NDArray[np.int8]
Int32Matrix = npt.NDArray[np.int32]
Float32Matrix = npt.NDArray[np.float32]


@dataclass(frozen=True)
class CatalystMatrix:
    """Session x security causal catalyst state through the close of each session."""

    tier1_on_session: BoolMatrix
    tier1_category: Int8Matrix
    latest_tier1_category: Int8Matrix
    sessions_since_tier1: Float32Matrix
    latest_tier1_row: Int32Matrix
    bearish_on_session: BoolMatrix
    has_bearish_3d: Float32Matrix
    report: dict[str, Any] = field(default_factory=dict)
    input_sha256: dict[str, str] = field(default_factory=dict)


def read_family_events(path: Path, development_end: date, *, unseal_final_test_period: bool = False) -> pd.DataFrame:
    """Read direct-issuer classified news events strictly before the end of ``development_end`` (ET)."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    if any(part.startswith(SEALED_DIRECTORY_PREFIX) for part in path.parts):
        raise SealedPeriodError(f"{path} belongs to a later-sealed catalyst directory and is not read")
    cutoff = acceptance_cutoff(development_end)
    table = cast(Any, pq).read_table(
        path,
        columns=list(FAMILY_COLUMNS),
        partitioning=None,
        filters=[("feature_available_at_utc", "<", cutoff.to_pydatetime())],
    )
    frame: pd.DataFrame = table.to_pandas()
    if frame.empty:
        return frame
    available = pd.DatetimeIndex(frame["feature_available_at_utc"]).tz_convert("UTC")
    event_avail = pd.DatetimeIndex(frame["event_available_at_utc"]).tz_convert("UTC")
    published = pd.DatetimeIndex(frame["published_at_utc"]).tz_convert("UTC")
    if available.isna().any() or bool((available >= cutoff).any()):
        raise SealedPeriodError(f"catalyst events at or after {cutoff} reached memory from {path}")
    if bool((event_avail > available).any()) or bool((published > available).any()):
        raise ValueError(f"{path} contains catalyst rows available before publication or event availability")
    if not bool(frame["relation_channel"].eq("direct_issuer").all()):
        raise ValueError(f"{path} contains non-direct-issuer relation channels")
    if not bool(frame["research_eligible"].fillna(False).astype(bool).all()):
        raise ValueError(f"{path} contains rows not marked research_eligible")
    return frame


def _pit_session_ticker_map(panel_root: Path, sessions: npt.NDArray[np.datetime64],
                            development_end: date) -> dict[tuple[int, str], str]:
    """Map (session_row_index, UPPER_TICKER) -> security_id using point-in-time membership only."""
    profile_dir = panel_root / PROFILE_DIRECTORY
    frames: list[pd.DataFrame] = []
    for part in sorted(profile_dir.glob("year=*/part.parquet")):
        year = int(part.parent.name.split("=", 1)[1])
        if year > development_end.year:
            continue
        table = cast(Any, pq).read_table(
            part,
            columns=list(PANEL_COLUMNS),
            partitioning=None,
            filters=[("session_date_et", "<=", development_end)],
        )
        frames.append(table.to_pandas())
    if not frames:
        return {}
    rows = pd.concat(frames, ignore_index=True)
    rows["session"] = np.asarray(pd.to_datetime(rows["session_date_et"]).to_numpy(), dtype="datetime64[D]")
    if len(rows) and rows["session"].max() > np.datetime64(development_end, "D"):
        raise SealedPeriodError(f"membership rows after {development_end} reached memory")
    in_cal = rows["session"].isin(sessions)
    rows = rows.loc[in_cal]
    known = (rows["membership_available_at_utc"] <= rows["decision_time_utc"]).fillna(False)
    key_count = rows.groupby(["ticker", "session"])["security_id"].transform("nunique")
    dup_sec = rows.duplicated(["security_id", "session"], keep=False)
    clean = rows.loc[known & key_count.eq(1) & ~dup_sec, ["session", "ticker", "security_id"]]
    session_idx = np.searchsorted(sessions, clean["session"].to_numpy().astype("datetime64[D]"))
    tickers = clean["ticker"].astype(str).str.upper().to_numpy()
    sec_ids = clean["security_id"].astype(str).to_numpy()
    return {(int(s_idx), str(sym)): str(sec) for s_idx, sym, sec in zip(session_idx, tickers, sec_ids, strict=True)}


def classify_event_polarity(event_family: str, matched_text: str) -> tuple[bool, int]:
    """Return (is_bearish_or_dilutive, tier1_positive_category_code).

    Returns ``(False, 0)`` if the matched text is retrospective noise or an unrecognized family.
    """
    text = matched_text or ""
    if _RETROSPECTIVE_NOISE.search(text):
        return False, 0
    if event_family == "offering" or _BEARISH_TEXT.search(text):
        return True, 0
    code = CATEGORY_CODES.get(event_family, 0)
    return False, code


def build_catalyst_matrix_from_frames(
    panel: ResearchPanel,
    *,
    news_events: pd.DataFrame,
    sec_releases: pd.DataFrame,
    pit_ticker_map: dict[tuple[int, str], str],
    input_sha256: dict[str, str] | None = None,
) -> CatalystMatrix:
    """Project point-in-time news and SEC 8-K release frames onto ``panel`` (sessions x securities)."""
    n_sessions, n_secs = len(panel.sessions), len(panel.security_ids)
    shape = (n_sessions, n_secs)
    col_of = {sec_id: idx for idx, sec_id in enumerate(panel.security_ids)}
    # Unique single-class CIK prefix map (only CIKs owned by a single security column in panel).
    cik_counts = Counter(sec_id.split(":")[1] for sec_id in panel.security_ids if sec_id.startswith("cik:"))
    unique_cik_col = {
        f"cik:{sec_id.split(':')[1]}": idx
        for idx, sec_id in enumerate(panel.security_ids)
        if sec_id.startswith("cik:") and sec_id.count(":") == 1 and cik_counts[sec_id.split(":")[1]] == 1
    }

    tier1_on = np.zeros(shape, dtype=bool)
    tier1_cat = np.zeros(shape, dtype=np.int8)
    bearish_on = np.zeros(shape, dtype=bool)
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    def resolve_column(session_idx: int, security_id: str, ticker: str) -> tuple[int, str]:
        upper = ticker.upper()
        pit_hit = pit_ticker_map.get((session_idx, upper))
        if pit_hit is not None and pit_hit in col_of:
            return col_of[pit_hit], "pit_session_ticker"
        if security_id in col_of:
            return col_of[security_id], "exact_security_id"
        if security_id.startswith("cik:"):
            prefix = f"cik:{security_id.split(':')[1]}"
            if prefix in unique_cik_col:
                return unique_cik_col[prefix], "unique_cik"
        return -1, "unmapped"

    if not news_events.empty:
        s_indices = event_rows(panel.sessions, news_events["feature_available_at_utc"])
        sec_ids = news_events["security_id"].astype(str).to_numpy()
        tickers = news_events["ticker"].fillna("").astype(str).to_numpy()
        families = news_events["event_family"].fillna("").astype(str).to_numpy()
        texts = news_events["matched_text"].fillna("").astype(str).to_numpy()
        for s_idx_raw, sec_id, sym, fam, txt in zip(s_indices, sec_ids, tickers, families, texts, strict=True):
            s_idx = int(s_idx_raw)
            if s_idx >= n_sessions:
                counts["news_after_last_panel_session"] += 1
                continue
            if _RETROSPECTIVE_NOISE.search(txt):
                counts["news_excluded_retrospective_noise"] += 1
                continue
            col, how = resolve_column(s_idx, str(sec_id), str(sym))
            counts[f"news_map_{how}"] += 1
            if col < 0:
                continue
            is_bearish, code = classify_event_polarity(str(fam), str(txt))
            if is_bearish:
                bearish_on[s_idx, col] = True
                counts["news_bearish_or_dilutive_events"] += 1
            elif code > 0:
                tier1_on[s_idx, col] = True
                current = int(tier1_cat[s_idx, col])
                if current == 0 or code < current:
                    tier1_cat[s_idx, col] = np.int8(code)
                counts["news_tier1_positive_events"] += 1
                category_counts[CODE_TO_CATEGORY[code]] += 1

    if not sec_releases.empty:
        sec_s_indices = event_rows(panel.sessions, sec_releases["available_at_utc"])
        sec_sids = sec_releases["security_id"].astype(str).to_numpy()
        for s_idx_raw, sec_id in zip(sec_s_indices, sec_sids, strict=True):
            s_idx = int(s_idx_raw)
            if s_idx >= n_sessions:
                counts["sec_after_last_panel_session"] += 1
                continue
            col = col_of.get(str(sec_id), -1)
            if col < 0 and str(sec_id).startswith("cik:"):
                col = unique_cik_col.get(f"cik:{str(sec_id).split(':')[1]}", -1)
            if col < 0:
                counts["sec_unmapped"] += 1
                continue
            tier1_on[s_idx, col] = True
            tier1_cat[s_idx, col] = np.int8(CATEGORY_CODES["earnings"])
            counts["sec_8k_earnings_events"] += 1
            category_counts["sec_8k_202"] += 1

    # Forward-propagate causal state across sessions (row t depends on rows <= t only).
    latest_cat = np.zeros(shape, dtype=np.int8)
    sessions_since = np.full(shape, np.nan, dtype=np.float32)
    latest_row = np.full(shape, -1, dtype=np.int32)
    cur_cat = np.zeros(n_secs, dtype=np.int8)
    cur_row = np.full(n_secs, -1, dtype=np.int32)
    for row in range(n_sessions):
        hit = tier1_on[row]
        cur_cat = np.where(hit, tier1_cat[row], cur_cat)
        cur_row = np.where(hit, np.int32(row), cur_row)
        age = row - cur_row
        active = (cur_row >= 0) & (age <= CATALYST_EXPIRY_SESSIONS)
        latest_cat[row] = np.where(active, cur_cat, np.int8(0))
        sessions_since[row] = np.where(active, age.astype(np.float32), np.float32(np.nan))
        latest_row[row] = np.where(active, cur_row, np.int32(-1))

    bearish_f = bearish_on.astype(np.float32)
    lag1 = np.vstack([np.zeros((1, n_secs), dtype=np.float32), bearish_f[:-1]])
    lag2 = np.vstack([np.zeros((2, n_secs), dtype=np.float32), bearish_f[:-2]])
    has_bearish_3d = np.where((bearish_f + lag1 + lag2) > 0, np.float32(1.0), np.float32(0.0))

    report: dict[str, Any] = {
        "news_rows_input": len(news_events),
        "sec_earnings_rows_input": len(sec_releases),
        "tier1_stock_sessions": int(tier1_on.sum()),
        "bearish_stock_sessions": int(bearish_on.sum()),
        "event_counts": dict(sorted(counts.items())),
        "tier1_events_by_category": dict(sorted(category_counts.items())),
    }
    return CatalystMatrix(
        tier1_on_session=tier1_on,
        tier1_category=tier1_cat,
        latest_tier1_category=latest_cat,
        sessions_since_tier1=sessions_since,
        latest_tier1_row=latest_row,
        bearish_on_session=bearish_on,
        has_bearish_3d=has_bearish_3d,
        report=report,
        input_sha256=dict(input_sha256 or {}),
    )


def load_catalyst_matrix(
    panel: ResearchPanel,
    *,
    panel_root: Path,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    filings_dir: Path = FILINGS_DIRECTORY,
    unseal_final_test_period: bool = False,
) -> CatalystMatrix:
    """Load retained Tier-1 Alpaca news and SEC 8-K catalysts through ``panel.development_end``."""
    end = panel.development_end
    check_development_end(end, unseal_final_test_period=unseal_final_test_period)
    early_events_path = early_news_dir / FAMILY_EVENTS_FILE
    later_events_path = later_news_dir / FAMILY_EVENTS_FILE
    filings_path = filings_dir / FILINGS_FILE
    hashes = {
        "early_news_family_events_sha256": file_sha256(early_events_path),
        "early_news_family_manifest_sha256": file_sha256(early_news_dir / FAMILY_MANIFEST_FILE),
        "later_news_family_events_sha256": file_sha256(later_events_path),
        "later_news_family_manifest_sha256": file_sha256(later_news_dir / FAMILY_MANIFEST_FILE),
        "sec_filings_parquet_sha256": file_sha256(filings_path),
        "sec_filings_manifest_sha256": file_sha256(filings_dir / FILINGS_MANIFEST),
    }
    early_df = read_family_events(early_events_path, end, unseal_final_test_period=unseal_final_test_period)
    later_df = read_family_events(later_events_path, end, unseal_final_test_period=unseal_final_test_period)
    news_df = pd.concat([early_df, later_df], ignore_index=True)
    filings_df = read_sec_filings(filings_path, end)
    releases_df, sec_report = earnings_releases(filings_df)
    del filings_df
    pit_map = _pit_session_ticker_map(panel_root, panel.sessions, end)
    matrix = build_catalyst_matrix_from_frames(
        panel,
        news_events=news_df,
        sec_releases=releases_df,
        pit_ticker_map=pit_map,
        input_sha256=hashes,
    )
    matrix.report["sec_release_summary"] = sec_report
    return matrix
