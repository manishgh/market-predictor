"""Step 5f (Option A): Zero-Fit Point-in-Time Alpaca `OPRA` Option-Chain Metrics for Forward Recording.

Computes three institutional, zero-fit option-market signals strictly as of `as_of_date`
(`> 2026-07-08`, zero access to the sealed `2024-05-29..2026-07-08` holdout window and zero
access to quotes or bars after `as_of_date` close):

1. **`SPY` & `QQQ` Constant-Maturity `30d / 90d` ATM IV Term Structure:**
   Variance-interpolated `30d` and `90d` ATM implied volatility (`spy_iv_30d`, `spy_iv_90d`,
   `spy_iv_term_30_90`, `qqq_iv_30d`, `qqq_iv_90d`, `qqq_iv_term_30_90`, `opra_backwardation`).
2. **Front-Expiry `ATM Straddle %` (`DTE in [2, 35]`) & Catalyst Shock-to-Expected-Move Ratio:**
   `atm_straddle_pct = (call_px + put_px) / spot_close` at the nearest-to-spot strike `K*`,
   and `catalyst_shock_to_straddle = |latest_catalyst_rx_1d| / atm_straddle_pct` (`> 1.0` flags
   catalyst reactions that exceeded the options-priced move).
3. **`25-Delta` Put–Call Implied Volatility Skew (`IV_25d_Put - IV_25d_Call`) & Falling-Knife Veto:**
   Safeguarded Black-Scholes IV & analytic delta inversion on `as_of_date` `OPRA` `1Day`
   VWAP/close bars (`DTE in [14, 65]`), `atm_iv_to_atr_ratio`, `falling_knife_put_skew_veto`
   (worst decile `>= 90th pct` and `put_call_skew_25d > 0.05` masking `score_reversal_short`),
   and `score_r2_opra_conditioned` (Step 5e's `R2` state machine driven by `spy_iv_term_30_90`).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import requests

from market_predictor.config import Settings
from market_predictor.core.runtime_memory import guard
from market_predictor.research_lane import indicators, sec_catalyst_model
from market_predictor.research_lane.panel import MARKET_TIMEZONE, ResearchPanel, SealedPeriodError
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.prospective_contracts import (
    OpraObservationPolicy,
    OpraRegimeState,
    extract_snapshot_opra_regime,
    resolve_opra_regime,
    resolve_xnys_session_clock,
)

HISTORICAL_SEALED_END: Final = date(2026, 7, 8)
ALPACA_OPTIONS_BARS_URL: Final = "https://data.alpaca.markets/v1beta1/options/bars"
ALPACA_STOCKS_BARS_URL: Final = "https://data.alpaca.markets/v2/stocks/bars"
DEFAULT_RISK_FREE_RATE: Final = 0.043
CLOSE_OBSERVATION_WINDOW_MINUTES: Final = 15
FIVE_MIN_BAR_SECONDS: Final = 300
MAX_SPOT_SKEW_SECONDS: Final = 300.0
MAX_CLOSE_STALENESS_SECONDS: Final = 900.0
_OCC_SYMBOL_RE: Final = re.compile(r"^([A-Z][A-Z0-9.]{0,5})\s*(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True, slots=True)
class OccContract:
    symbol: str
    underlying: str
    expiration: date
    option_type: str  # "call" or "put"
    strike: float


@dataclass(frozen=True, slots=True)
class SynchronizedContractObservation:
    """Closing-window `5Min` option close paired with timestamp-aligned `5Min` underlying spot (`R10`)."""

    contract: OccContract
    option_price: float
    trade_count: int
    underlying_spot: float
    option_bar_start_utc: datetime
    underlying_bar_start_utc: datetime
    spot_skew_seconds: float
    close_staleness_seconds: float


def _opt_finite_float(val: object) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        fv = float(cast(Any, val))
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def parse_occ_option_symbol(occ_symbol: str) -> OccContract | None:
    """Decode standard 21-char OCC option symbol (`CROX261218C00100000`) into structured fields."""
    m = _OCC_SYMBOL_RE.match(occ_symbol.strip().upper())
    if m is None:
        return None
    und, yymmdd, cp, strike_8 = m.groups()
    try:
        exp = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    strike = int(strike_8) / 1000.0
    if strike <= 0.0:
        return None
    return OccContract(
        symbol=occ_symbol.strip().upper(),
        underlying=und,
        expiration=exp,
        option_type="call" if cp == "C" else "put",
        strike=strike,
    )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price_and_delta(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    *,
    option_type: str,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[float, float]:
    """Analytic Black-Scholes European price and delta (`(0, 1)` for call, `(-1, 0)` for put)."""
    if spot <= 0.0 or strike <= 0.0 or t_years <= 0.0 or sigma <= 0.0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df = math.exp(-r * t_years)
    if option_type == "call":
        nd1 = _norm_cdf(d1)
        price = spot * nd1 - strike * df * _norm_cdf(d2)
        return price, nd1
    nd1 = _norm_cdf(d1)
    price = strike * df * _norm_cdf(-d2) - spot * (1.0 - nd1)
    return price, nd1 - 1.0


def solve_iv_and_delta(
    market_price: float,
    spot: float,
    strike: float,
    t_years: float,
    *,
    option_type: str,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[float | None, float | None]:
    """Safeguarded Newton-Raphson + bisection Black-Scholes IV solver over `[0.01, 5.00]`."""
    if market_price <= 0.0 or spot <= 0.0 or strike <= 0.0 or t_years <= 1.0 / 3650.0:
        return None, None
    df = math.exp(-r * t_years)
    if option_type == "call":
        intrinsic = max(0.0, spot - strike * df)
        upper = spot
    else:
        intrinsic = max(0.0, strike * df - spot)
        upper = strike * df

    if market_price <= intrinsic + 1e-5 or market_price >= upper - 1e-5:
        return None, None

    lo, hi = 0.01, 5.00
    p_lo, _ = bs_price_and_delta(spot, strike, t_years, lo, option_type=option_type, r=r)
    p_hi, _ = bs_price_and_delta(spot, strike, t_years, hi, option_type=option_type, r=r)
    if market_price < p_lo or market_price > p_hi:
        return None, None

    sigma = min(max(math.sqrt(2.0 * math.pi / t_years) * (market_price / spot), lo), hi)
    sqrt_t = math.sqrt(t_years)
    for _ in range(36):
        price, delta = bs_price_and_delta(spot, strike, t_years, sigma, option_type=option_type, r=r)
        diff = price - market_price
        if abs(diff) <= 1e-7:
            return sigma, delta
        if diff < 0.0:
            lo = max(lo, sigma)
        else:
            hi = min(hi, sigma)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
        vega = spot * _norm_pdf(d1) * sqrt_t
        if vega > 1e-8:
            cand = sigma - diff / vega
            sigma = cand if lo < cand < hi else 0.5 * (lo + hi)
        else:
            sigma = 0.5 * (lo + hi)

    _, final_delta = bs_price_and_delta(spot, strike, t_years, sigma, option_type=option_type, r=r)
    return sigma, final_delta


def _interpolate_constant_maturity_iv(
    atm_by_dte: list[tuple[int, float]],
    target_dte: int,
    *,
    min_dte: int,
    max_dte: int,
) -> float | None:
    """Variance-linear interpolation (`sigma^2 * T`) for constant-maturity ATM IV (e.g., 30d or 90d)."""
    pts = sorted((d, iv) for d, iv in atm_by_dte if min_dte <= d <= max_dte and np.isfinite(iv) and iv > 0.0)
    if not pts:
        return None
    below = [p for p in pts if p[0] <= target_dte]
    above = [p for p in pts if p[0] >= target_dte]
    if below and above:
        d1, iv1 = below[-1]
        d2, iv2 = above[0]
        if d1 == d2:
            return float(iv1)
        t1, t2, tt = d1 / 365.0, d2 / 365.0, target_dte / 365.0
        var1, var2 = iv1 * iv1 * t1, iv2 * iv2 * t2
        w = (tt - t1) / (t2 - t1)
        var_t = max(1e-10, var1 + w * (var2 - var1))
        return float(math.sqrt(var_t / tt))
    nearest = min(pts, key=lambda p: abs(p[0] - target_dte))
    return float(nearest[1])


def compute_symbol_option_metrics(
    contracts_with_prices: list[tuple[OccContract, float, int]],
    *,
    spot_close: float,
    as_of_date: date,
    atr_pct_14: float | None = None,
    latest_catalyst_rx_1d: float | None = None,
) -> dict[str, Any]:
    """Compute front-week `ATM Straddle %`, `25d` Put-Call IV Skew, and `30d/90d` ATM IV from `as_of_date` bars."""
    empty: dict[str, Any] = {
        "atm_straddle_pct": None,
        "atm_straddle_dte": None,
        "iv_30d": None,
        "iv_90d": None,
        "iv_term_30_90": None,
        "iv_put_25d": None,
        "iv_call_25d": None,
        "put_call_skew_25d": None,
        "atm_iv_to_atr_ratio": None,
        "catalyst_shock_to_straddle": None,
    }
    if spot_close <= 0.0 or not contracts_with_prices:
        return empty

    by_exp_strike: dict[tuple[date, float], dict[str, float]] = {}
    solved_rows: list[tuple[int, str, float, float, float]] = []  # (dte, type, strike, iv, delta)

    for occ, px, n_trades in contracts_with_prices:
        if px <= 0.0 or n_trades < 1:
            continue
        dte = (occ.expiration - as_of_date).days
        if dte < 2 or dte > 130:
            continue
        by_exp_strike.setdefault((occ.expiration, occ.strike), {})[occ.option_type] = px
        iv, delta = solve_iv_and_delta(px, spot_close, occ.strike, dte / 365.0, option_type=occ.option_type)
        if iv is not None and delta is not None:
            solved_rows.append((dte, occ.option_type, occ.strike, iv, delta))

    # 1. Front-expiry ATM Straddle % (`DTE in [2, 35]`, nearest expiration with both Call & Put within 8% of spot)
    straddle_pct: float | None = None
    straddle_dte: int | None = None
    front_exps = sorted({exp for (exp, _) in by_exp_strike if 2 <= (exp - as_of_date).days <= 35})
    for exp in front_exps:
        both_strikes = [
            (abs(k - spot_close), k, sides["call"], sides["put"])
            for (e_k, k), sides in by_exp_strike.items()
            if e_k == exp and "call" in sides and "put" in sides and abs(k / spot_close - 1.0) <= 0.08
        ]
        if both_strikes:
            both_strikes.sort(key=lambda item: item[0])
            _, _, c_px, p_px = both_strikes[0]
            straddle_pct = float((c_px + p_px) / spot_close)
            straddle_dte = (exp - as_of_date).days
            break

    # 2. Constant-maturity `30d` & `90d` ATM IV across expirations
    atm_by_exp: dict[int, list[tuple[float, float]]] = {}
    for dte, _, k, iv, _ in solved_rows:
        if abs(k / spot_close - 1.0) <= 0.08:
            atm_by_exp.setdefault(dte, []).append((abs(k - spot_close), iv))
    atm_dte_iv: list[tuple[int, float]] = []
    for dte, cand_ivs in atm_by_exp.items():
        cand_ivs.sort(key=lambda x: x[0])
        top_k = [v for _, v in cand_ivs[:2]]
        atm_dte_iv.append((dte, float(sum(top_k) / len(top_k))))

    iv_30d = _interpolate_constant_maturity_iv(atm_dte_iv, 30, min_dte=10, max_dte=55)
    iv_90d = _interpolate_constant_maturity_iv(atm_dte_iv, 90, min_dte=55, max_dte=130)
    iv_term = float(iv_30d / iv_90d) if (iv_30d is not None and iv_90d is not None and iv_90d > 0.0) else None

    # 3. `25-Delta` Put–Call IV Skew (`DTE in [14, 65]`)
    put_cands = [
        (abs(delta - (-0.25)), abs(dte - 30), iv)
        for dte, otype, _, iv, delta in solved_rows
        if otype == "put" and 14 <= dte <= 65 and -0.45 <= delta <= -0.10
    ]
    call_cands = [
        (abs(delta - 0.25), abs(dte - 30), iv)
        for dte, otype, _, iv, delta in solved_rows
        if otype == "call" and 14 <= dte <= 65 and 0.10 <= delta <= 0.45
    ]
    iv_put_25 = float(min(put_cands, key=lambda x: (x[0], x[1]))[2]) if put_cands else None
    iv_call_25 = float(min(call_cands, key=lambda x: (x[0], x[1]))[2]) if call_cands else None
    skew_25 = float(iv_put_25 - iv_call_25) if (iv_put_25 is not None and iv_call_25 is not None) else None

    iv_to_atr: float | None = None
    if iv_30d is not None and atr_pct_14 is not None and np.isfinite(atr_pct_14) and atr_pct_14 > 1e-6:
        daily_iv = iv_30d / math.sqrt(252.0)
        iv_to_atr = float(daily_iv / atr_pct_14)

    shock_ratio: float | None = None
    if (
        straddle_pct is not None
        and straddle_pct > 1e-5
        and latest_catalyst_rx_1d is not None
        and np.isfinite(latest_catalyst_rx_1d)
    ):
        shock_ratio = float(abs(latest_catalyst_rx_1d) / straddle_pct)

    return {
        "atm_straddle_pct": straddle_pct,
        "atm_straddle_dte": straddle_dte,
        "iv_30d": iv_30d,
        "iv_90d": iv_90d,
        "iv_term_30_90": iv_term,
        "iv_put_25d": iv_put_25,
        "iv_call_25d": iv_call_25,
        "put_call_skew_25d": skew_25,
        "atm_iv_to_atr_ratio": iv_to_atr,
        "catalyst_shock_to_straddle": shock_ratio,
    }


def _fetch_contracts_for_symbol(
    base_trading_url: str,
    headers: dict[str, str],
    symbol: str,
    spot: float,
    as_of_date: date,
    *,
    max_dte: int,
) -> list[OccContract]:
    """Fetch targeted front-ATM, 1M ATM+25d-skew, and 3M ATM OCC contracts without 0DTE/weekly truncation."""
    contracts_url = f"{base_trading_url.rstrip('/')}/v2/options/contracts"
    is_index = symbol in ("SPY", "QQQ")
    windows: list[tuple[int, int, float, float]] = [
        (2, min(20, max_dte), 0.04 if is_index else 0.08, 1.50),
        (21, min(55, max_dte), 0.10 if is_index else 0.22, 2.50),
    ]
    if max_dte >= 85:
        windows.append((72, min(max_dte, 125), 0.04 if is_index else 0.08, 1.50))

    found: dict[str, OccContract] = {}
    for d_lo, d_hi, pct_w, d_pad in windows:
        if d_hi < d_lo:
            continue
        exp_gte = (as_of_date + timedelta(days=d_lo)).isoformat()
        exp_lte = (as_of_date + timedelta(days=d_hi)).isoformat()
        pad = max(spot * pct_w, d_pad)
        s_lo = f"{max(0.5, spot - pad):.2f}"
        s_hi = f"{spot + pad:.2f}"
        win_added = 0
        for status_val in ("active", "inactive"):
            resp = requests.get(
                contracts_url,
                params={
                    "underlying_symbols": symbol,
                    "status": status_val,
                    "expiration_date_gte": exp_gte,
                    "expiration_date_lte": exp_lte,
                    "strike_price_gte": s_lo,
                    "strike_price_lte": s_hi,
                    "limit": 1000,
                },
                headers=headers,
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            raw_list = payload.get("option_contracts", []) if isinstance(payload, dict) else []
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                occ = parse_occ_option_symbol(str(item.get("symbol") or ""))
                if occ is not None and occ.underlying == symbol:
                    found[occ.symbol] = occ
                    win_added += 1
            if status_val == "active" and win_added >= 4:
                break

    selected: list[OccContract] = []
    seen: set[str] = set()

    def _take(cands: list[OccContract], limit_n: int) -> None:
        for c in cands:
            if c.symbol not in seen:
                selected.append(c)
                seen.add(c.symbol)
                if len(selected) >= 40:
                    return
                limit_n -= 1
                if limit_n <= 0:
                    return

    front = sorted(
        [c for c in found.values() if 2 <= (c.expiration - as_of_date).days <= 20],
        key=lambda c: (abs(c.strike - spot), c.expiration),
    )
    _take(front, 10)

    mid_all = [c for c in found.values() if 21 <= (c.expiration - as_of_date).days <= 65]
    mid_atm = sorted(mid_all, key=lambda c: (abs(c.strike - spot), abs((c.expiration - as_of_date).days - 31)))
    _take(mid_atm, 6)

    put_target = spot * (0.955 if is_index else 0.90)
    mid_puts = sorted(
        [c for c in mid_all if c.option_type == "put" and c.strike <= spot],
        key=lambda c: (abs(c.strike - put_target), abs((c.expiration - as_of_date).days - 31)),
    )
    _take(mid_puts, 6)

    call_target = spot * (1.045 if is_index else 1.10)
    mid_calls = sorted(
        [c for c in mid_all if c.option_type == "call" and c.strike >= spot],
        key=lambda c: (abs(c.strike - call_target), abs((c.expiration - as_of_date).days - 31)),
    )
    _take(mid_calls, 6)

    back = sorted(
        [c for c in found.values() if 66 <= (c.expiration - as_of_date).days <= 130],
        key=lambda c: (abs(c.strike - spot), abs((c.expiration - as_of_date).days - 91)),
    )
    _take(back, 8)
    return selected


def _fetch_closing_option_5min_bars(
    headers: dict[str, str],
    contracts: list[OccContract],
    as_of_date: date,
    *,
    session_close_utc: datetime,
    unseal_final_test_period: bool = False,
) -> dict[str, tuple[float, int, datetime]]:
    """Fetch latest closing-window `5Min` option `close` (`[session_close_utc - 15m, session_close_utc]`, zero post-close trades)."""
    if as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"OPRA bars refused on or before {HISTORICAL_SEALED_END.isoformat()}")
    if not contracts:
        return {}

    win_start_utc = session_close_utc - timedelta(minutes=CLOSE_OBSERVATION_WINDOW_MINUTES)
    query_end_utc = session_close_utc - timedelta(seconds=1)
    out: dict[str, tuple[float, int, datetime]] = {}
    syms = [c.symbol for c in contracts]

    for off in range(0, len(syms), 80):
        chunk = syms[off : off + 80]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": "5Min",
                "start": win_start_utc.isoformat(),
                "end": query_end_utc.isoformat(),
                "limit": 1000,
            }
            if page_token:
                params["page_token"] = page_token
            resp = requests.get(ALPACA_OPTIONS_BARS_URL, params=params, headers=headers, timeout=25)
            if resp.status_code != 200:
                break
            js = resp.json()
            bars_map = js.get("bars", {}) if isinstance(js, dict) else {}
            for sym_k, b_list in bars_map.items():
                if not isinstance(b_list, list) or not b_list:
                    continue
                sym_up = str(sym_k).upper()
                for b in b_list:
                    b_ts = pd.Timestamp(str(b["t"])).tz_convert("UTC").to_pydatetime()
                    b_day = pd.Timestamp(b_ts).tz_convert(MARKET_TIMEZONE).date()
                    if b_day > as_of_date:
                        raise SealedPeriodError(f"OPRA 5Min bar after as_of_date ({b_day} > {as_of_date})")
                    if b_day != as_of_date:
                        continue
                    if b_ts < win_start_utc or (b_ts + timedelta(seconds=FIVE_MIN_BAR_SECONDS)) > session_close_utc:
                        continue
                    px = float(b.get("c") or 0.0)
                    n_tr = int(b.get("n") or b.get("v") or 0)
                    if px <= 0.0 or n_tr < 1:
                        continue
                    prev = out.get(sym_up)
                    if prev is None or b_ts >= prev[2]:
                        out[sym_up] = (px, n_tr, b_ts)
            page_token = js.get("next_page_token") if isinstance(js, dict) else None
            if not page_token:
                break
    return out


def _fetch_closing_underlying_5min_bars(
    headers: dict[str, str],
    symbols: list[str],
    as_of_date: date,
    *,
    session_close_utc: datetime,
    stock_feed: str = "sip",
    unseal_final_test_period: bool = False,
) -> dict[str, dict[datetime, float]]:
    """Fetch closing-window `5Min` unadjusted underlying `close` bars (`[session_close_utc - 15m, session_close_utc]`)."""
    if as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"underlying 5Min bars refused on or before {HISTORICAL_SEALED_END.isoformat()}")
    uniq = sorted({s.upper().strip() for s in symbols if s.strip()})
    if not uniq:
        return {}

    win_start_utc = session_close_utc - timedelta(minutes=CLOSE_OBSERVATION_WINDOW_MINUTES)
    query_end_utc = session_close_utc - timedelta(seconds=1)
    out: dict[str, dict[datetime, float]] = {}

    for off in range(0, len(uniq), 50):
        chunk = uniq[off : off + 50]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": "5Min",
                "start": win_start_utc.isoformat(),
                "end": query_end_utc.isoformat(),
                "feed": stock_feed,
                "adjustment": "raw",
                "limit": 10000,
            }
            if page_token:
                params["page_token"] = page_token
            resp = requests.get(ALPACA_STOCKS_BARS_URL, params=params, headers=headers, timeout=25)
            if resp.status_code != 200:
                break
            js = resp.json()
            bars_map = js.get("bars", {}) if isinstance(js, dict) else {}
            for sym_k, b_list in bars_map.items():
                if not isinstance(b_list, list) or not b_list:
                    continue
                sym_up = str(sym_k).upper()
                for b in b_list:
                    b_ts = pd.Timestamp(str(b["t"])).tz_convert("UTC").to_pydatetime()
                    b_day = pd.Timestamp(b_ts).tz_convert(MARKET_TIMEZONE).date()
                    if b_day > as_of_date:
                        raise SealedPeriodError(f"underlying 5Min bar after as_of_date ({b_day} > {as_of_date})")
                    if b_day != as_of_date:
                        continue
                    if b_ts < win_start_utc or (b_ts + timedelta(seconds=FIVE_MIN_BAR_SECONDS)) > session_close_utc:
                        continue
                    px = float(b.get("c") or 0.0)
                    if px > 0.0:
                        out.setdefault(sym_up, {})[b_ts] = px
            page_token = js.get("next_page_token") if isinstance(js, dict) else None
            if not page_token:
                break
    return out


def pair_synchronized_close_contracts(
    contracts: list[OccContract],
    option_5m_map: Mapping[str, tuple[float, int, datetime]],
    underlying_5m_map: Mapping[str, Mapping[datetime, float]],
    *,
    session_close_utc: datetime,
) -> list[SynchronizedContractObservation]:
    """Pair each contract's closing-window `5Min` option close with timestamp-matched `5Min` underlying spot (`<= 5m` skew)."""
    paired: list[SynchronizedContractObservation] = []
    for c in contracts:
        opt_hit = option_5m_map.get(c.symbol)
        if opt_hit is None:
            continue
        opt_px, n_tr, opt_ts = opt_hit
        und_bars = underlying_5m_map.get(c.underlying)
        if not und_bars:
            continue
        best_u_ts = min(und_bars.keys(), key=lambda u_ts: abs((u_ts - opt_ts).total_seconds()))
        skew_sec = abs((best_u_ts - opt_ts).total_seconds())
        staleness_sec = (session_close_utc - (opt_ts + timedelta(seconds=FIVE_MIN_BAR_SECONDS))).total_seconds()
        if skew_sec > MAX_SPOT_SKEW_SECONDS or staleness_sec < 0.0 or staleness_sec > MAX_CLOSE_STALENESS_SECONDS:
            continue
        u_spot = float(und_bars[best_u_ts])
        if u_spot <= 0.0 or opt_px <= 0.0 or n_tr < 1:
            continue
        paired.append(
            SynchronizedContractObservation(
                contract=c,
                option_price=opt_px,
                trade_count=n_tr,
                underlying_spot=u_spot,
                option_bar_start_utc=opt_ts,
                underlying_bar_start_utc=best_u_ts,
                spot_skew_seconds=float(skew_sec),
                close_staleness_seconds=float(staleness_sec),
            )
        )
    return paired


def compute_symbol_option_metrics_synchronized(
    observations: list[SynchronizedContractObservation],
    *,
    as_of_date: date,
    atr_pct_14: float | None = None,
    latest_catalyst_rx_1d: float | None = None,
) -> dict[str, Any]:
    """Compute option metrics strictly from timestamp-synchronized closing `5Min` option & underlying pairs (`R10`)."""
    empty: dict[str, Any] = {
        "atm_straddle_pct": None,
        "atm_straddle_dte": None,
        "iv_30d": None,
        "iv_90d": None,
        "iv_term_30_90": None,
        "iv_put_25d": None,
        "iv_call_25d": None,
        "put_call_skew_25d": None,
        "atm_iv_to_atr_ratio": None,
        "catalyst_shock_to_straddle": None,
        "synchronized_contracts_used": 0,
        "latest_option_bar_utc": None,
        "max_spot_skew_seconds": None,
        "max_close_staleness_seconds": None,
    }
    if not observations:
        return empty

    by_exp_strike: dict[tuple[date, float], dict[str, SynchronizedContractObservation]] = {}
    solved_rows: list[tuple[int, str, float, float, float, float]] = []  # (dte, type, strike, spot, iv, delta)
    used: list[SynchronizedContractObservation] = []

    for obs in observations:
        occ = obs.contract
        if obs.option_price <= 0.0 or obs.trade_count < 1 or obs.underlying_spot <= 0.0:
            continue
        dte = (occ.expiration - as_of_date).days
        if dte < 2 or dte > 130:
            continue
        used.append(obs)
        by_exp_strike.setdefault((occ.expiration, occ.strike), {})[occ.option_type] = obs
        iv, delta = solve_iv_and_delta(
            obs.option_price,
            obs.underlying_spot,
            occ.strike,
            dte / 365.0,
            option_type=occ.option_type,
        )
        if iv is not None and delta is not None:
            solved_rows.append((dte, occ.option_type, occ.strike, obs.underlying_spot, iv, delta))

    if not used:
        return empty

    straddle_pct: float | None = None
    straddle_dte: int | None = None
    front_exps = sorted({exp for (exp, _) in by_exp_strike if 2 <= (exp - as_of_date).days <= 35})
    for exp in front_exps:
        both_strikes: list[tuple[float, float, float, float]] = []
        for (e_k, k), sides in by_exp_strike.items():
            if e_k != exp or "call" not in sides or "put" not in sides:
                continue
            c_obs = sides["call"]
            p_obs = sides["put"]
            if abs((c_obs.option_bar_start_utc - p_obs.option_bar_start_utc).total_seconds()) > MAX_SPOT_SKEW_SECONDS:
                continue
            avg_spot = 0.5 * (c_obs.underlying_spot + p_obs.underlying_spot)
            if avg_spot > 0.0 and abs(k / avg_spot - 1.0) <= 0.08:
                both_strikes.append((abs(k - avg_spot), k, c_obs.option_price + p_obs.option_price, avg_spot))
        if both_strikes:
            both_strikes.sort(key=lambda item: item[0])
            _, _, tot_px, avg_sp = both_strikes[0]
            straddle_pct = float(tot_px / avg_sp)
            straddle_dte = (exp - as_of_date).days
            break

    atm_by_exp: dict[int, list[tuple[float, float]]] = {}
    for dte, _, k, sp_row, iv, _ in solved_rows:
        if sp_row > 0.0 and abs(k / sp_row - 1.0) <= 0.08:
            atm_by_exp.setdefault(dte, []).append((abs(k - sp_row), iv))
    atm_dte_iv: list[tuple[int, float]] = []
    for dte, cand_ivs in atm_by_exp.items():
        cand_ivs.sort(key=lambda x: x[0])
        top_k = [v for _, v in cand_ivs[:2]]
        atm_dte_iv.append((dte, float(sum(top_k) / len(top_k))))

    iv_30d = _interpolate_constant_maturity_iv(atm_dte_iv, 30, min_dte=10, max_dte=55)
    iv_90d = _interpolate_constant_maturity_iv(atm_dte_iv, 90, min_dte=55, max_dte=130)
    iv_term = float(iv_30d / iv_90d) if (iv_30d is not None and iv_90d is not None and iv_90d > 0.0) else None

    put_cands = [
        (abs(delta - (-0.25)), abs(dte - 30), iv)
        for dte, otype, _, _, iv, delta in solved_rows
        if otype == "put" and 14 <= dte <= 65 and -0.45 <= delta <= -0.10
    ]
    call_cands = [
        (abs(delta - 0.25), abs(dte - 30), iv)
        for dte, otype, _, _, iv, delta in solved_rows
        if otype == "call" and 14 <= dte <= 65 and 0.10 <= delta <= 0.45
    ]
    iv_put_25 = float(min(put_cands, key=lambda x: (x[0], x[1]))[2]) if put_cands else None
    iv_call_25 = float(min(call_cands, key=lambda x: (x[0], x[1]))[2]) if call_cands else None
    skew_25 = float(iv_put_25 - iv_call_25) if (iv_put_25 is not None and iv_call_25 is not None) else None

    iv_to_atr: float | None = None
    if iv_30d is not None and atr_pct_14 is not None and np.isfinite(atr_pct_14) and atr_pct_14 > 1e-6:
        daily_iv = iv_30d / math.sqrt(252.0)
        iv_to_atr = float(daily_iv / atr_pct_14)

    shock_ratio: float | None = None
    if (
        straddle_pct is not None
        and straddle_pct > 1e-5
        and latest_catalyst_rx_1d is not None
        and np.isfinite(latest_catalyst_rx_1d)
    ):
        shock_ratio = float(abs(latest_catalyst_rx_1d) / straddle_pct)

    return {
        "atm_straddle_pct": straddle_pct,
        "atm_straddle_dte": straddle_dte,
        "iv_30d": iv_30d,
        "iv_90d": iv_90d,
        "iv_term_30_90": iv_term,
        "iv_put_25d": iv_put_25,
        "iv_call_25d": iv_call_25,
        "put_call_skew_25d": skew_25,
        "atm_iv_to_atr_ratio": iv_to_atr,
        "catalyst_shock_to_straddle": shock_ratio,
        "synchronized_contracts_used": len(used),
        "latest_option_bar_utc": max(o.option_bar_start_utc for o in used).isoformat(),
        "max_spot_skew_seconds": float(max(o.spot_skew_seconds for o in used)),
        "max_close_staleness_seconds": float(max(o.close_staleness_seconds for o in used)),
    }


def _fetch_session_option_bars(
    headers: dict[str, str],
    contracts: list[OccContract],
    as_of_date: date,
    *,
    unseal_final_test_period: bool = False,
) -> dict[str, tuple[float, int]]:
    """Fetch `as_of_date` `1Day` `OPRA` daily-VWAP proxy bars (`vwap_proxy_*` diagnostic columns only, `R10`)."""
    if as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"OPRA bars refused on or before {HISTORICAL_SEALED_END.isoformat()}")
    if not contracts:
        return {}

    start_iso = datetime.combine(as_of_date, time(0, 0), tzinfo=UTC).isoformat()
    end_iso = datetime.combine(as_of_date, time(23, 59, 59), tzinfo=UTC).isoformat()
    out: dict[str, tuple[float, int]] = {}
    syms = [c.symbol for c in contracts]

    for off in range(0, len(syms), 80):
        chunk = syms[off : off + 80]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": start_iso,
                "end": end_iso,
                "limit": 1000,
            }
            if page_token:
                params["page_token"] = page_token
            resp = requests.get(ALPACA_OPTIONS_BARS_URL, params=params, headers=headers, timeout=25)
            if resp.status_code != 200:
                break
            js = resp.json()
            bars_map = js.get("bars", {}) if isinstance(js, dict) else {}
            for sym_k, b_list in bars_map.items():
                if not isinstance(b_list, list) or not b_list:
                    continue
                for b in b_list:
                    b_day = pd.Timestamp(str(b["t"])).tz_convert(MARKET_TIMEZONE).date()
                    if b_day > as_of_date:
                        raise SealedPeriodError(f"OPRA bar after as_of_date ({b_day} > {as_of_date})")
                    if b_day == as_of_date:
                        px = float(b.get("vw") or b.get("c") or 0.0)
                        n_tr = int(b.get("n") or b.get("v") or 1)
                        if px > 0.0:
                            out[str(sym_k).upper()] = (px, max(n_tr, 1))
            page_token = js.get("next_page_token") if isinstance(js, dict) else None
            if not page_token:
                break
    return out


def collect_live_opra_options_overlay(
    panel: ResearchPanel,
    scores_df: pd.DataFrame,
    settings: Settings,
    *,
    options_top_k: int = 100,
    opra_observation_policy: OpraObservationPolicy = "synchronized_close_only",
    unseal_final_test_period: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute point-in-time `OPRA` `SPY`/`QQQ` term structure, `ATM Straddle %`, `25d` Skew & zero-fit vetoes.

    Under `opra_observation_policy="synchronized_close_only"` (default), primary IV/skew/straddle and
    benchmark regime fields use ONLY closing-window `5Min` option closes paired with timestamp-matched
    `5Min` underlying spot (`<= 5m` skew, `<= 15m` before `XNYS` close). Daily-average `1Day` `vw`
    bars are quarantined in `vwap_proxy_*` diagnostic columns (`R10`), and missing/indeterminate index
    ratios preserve `opra_regime="unknown"` (`opra_backwardation=None`) (`R7`).
    """
    scored_day = date.fromisoformat(str(scores_df["session_date"].iloc[0]))
    if scored_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"cannot collect OPRA option metrics on or before {HISTORICAL_SEALED_END.isoformat()}")

    enriched = scores_df.copy()
    if not settings.has_alpaca:
        for col in (
            "atm_straddle_pct", "put_call_skew_25d", "atm_iv_30d", "atm_iv_to_atr_ratio",
            "catalyst_shock_to_straddle", "vwap_proxy_atm_straddle_pct", "vwap_proxy_atm_iv_30d",
            "vwap_proxy_put_call_skew_25d", "option_observation_kind",
            "score_reversal_skew_vetoed", "score_r2_opra_conditioned",
            "score_mid_small_bn_sec_xgb_opra_10d", "score_mid_small_bn_sec_xgb_opra_21d",
        ):
            enriched[col] = None
        enriched["falling_knife_put_skew_veto"] = False
        return enriched, pd.DataFrame(), {
            "opra_enabled": False,
            "option_observation_policy": opra_observation_policy,
            "benchmark_observation_kind": "not_collected",
            "opra_regime": "unknown",
            "opra_backwardation": None,
        }

    clock = resolve_xnys_session_clock(scored_day)
    close_utc = clock.session_close_utc
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key_id or "",
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret_value or "",
    }
    ind = indicators.compute_indicators(panel)
    atr_last = ind["atr_pct_14"][-1]
    col_of = {sid: idx for idx, sid in enumerate(panel.security_ids)}

    rank_col = (
        "score_mid_small_bn_sec_xgb_10d"
        if "score_mid_small_bn_sec_xgb_10d" in enriched.columns and enriched["score_mid_small_bn_sec_xgb_10d"].notna().any()
        else (
            "score_beta_neutral_price_12"
            if "score_beta_neutral_price_12" in enriched.columns and enriched["score_beta_neutral_price_12"].notna().any()
            else "score_price_equal_12"
        )
    )
    ordered = enriched.sort_values(rank_col, ascending=False, na_position="last")
    rev_top = enriched.sort_values("score_reversal_short", ascending=False, na_position="last")["security_id"].head(
        max(1, options_top_k // 2)
    )
    cand_sids = list(dict.fromkeys([
        *ordered["security_id"].head(max(1, options_top_k)).astype(str).tolist(),
        *rev_top.astype(str).tolist(),
    ]))[: max(1, options_top_k)]

    spy_spot = float(panel.spy_close[-1])
    qqq_spot = float(panel.qqq_close[-1])

    contracts_by_sym: dict[str, list[OccContract]] = {}
    spot_by_sym: dict[str, float] = {"SPY": spy_spot, "QQQ": qqq_spot}
    for bench_sym, b_spot in (("SPY", spy_spot), ("QQQ", qqq_spot)):
        if np.isfinite(b_spot) and b_spot > 0.0:
            contracts_by_sym[bench_sym] = _fetch_contracts_for_symbol(
                settings.alpaca_trading_base_url, headers, bench_sym, b_spot, scored_day, max_dte=125,
            )

    tk_by_sid_map: dict[str, str] = {}
    if "ticker" in enriched.columns:
        for _, r_tk in enriched[["security_id", "ticker"]].iterrows():
            s_k = str(r_tk["security_id"])
            t_k = str(r_tk["ticker"] or "").upper().strip()
            if t_k:
                tk_by_sid_map[s_k] = t_k

    sid_by_tk: dict[str, str] = {}
    for idx_s, sid in enumerate(cand_sids):
        tk = tk_by_sid_map.get(sid) or (sid.split(":", 1)[1] if ":" in sid else sid)
        c_idx = col_of.get(sid, -1)
        if c_idx < 0:
            continue
        sp_val = float(panel.close[-1, c_idx])
        if not np.isfinite(sp_val) or sp_val <= 0.0:
            continue
        sid_by_tk[tk] = sid
        spot_by_sym[tk] = sp_val
        contracts_by_sym[tk] = _fetch_contracts_for_symbol(
            settings.alpaca_trading_base_url, headers, tk, sp_val, scored_day, max_dte=70,
        )
        if (idx_s + 1) % 25 == 0:
            guard("opra contracts query")

    all_contracts = [c for clist in contracts_by_sym.values() for c in clist]
    underlying_syms = sorted(contracts_by_sym.keys())

    # Synchronized closing `5Min` option closes + timestamp-matched `5Min` underlying spot (`R10`)
    opt_5m_map = _fetch_closing_option_5min_bars(
        headers,
        all_contracts,
        scored_day,
        session_close_utc=close_utc,
        unseal_final_test_period=unseal_final_test_period,
    )
    und_5m_map = _fetch_closing_underlying_5min_bars(
        headers,
        underlying_syms,
        scored_day,
        session_close_utc=close_utc,
        stock_feed=settings.alpaca_stock_feed,
        unseal_final_test_period=unseal_final_test_period,
    )
    # Daily `1Day` VWAP proxy bars (diagnostic `vwap_proxy_*` only unless explicitly authorized)
    vwap_px_map = _fetch_session_option_bars(
        headers, all_contracts, scored_day, unseal_final_test_period=unseal_final_test_period,
    )
    guard("opra bars fetched")

    use_proxy_for_primary = opra_observation_policy == "allow_daily_vwap_proxy"

    # 1. Benchmark SPY & QQQ constant-maturity 30d/90d IV term structure (synchronized vs daily-VWAP proxy)
    sync_bench_metrics: dict[str, dict[str, Any]] = {}
    vwap_bench_metrics: dict[str, dict[str, Any]] = {}
    for b_sym in ("SPY", "QQQ"):
        b_contracts = contracts_by_sym.get(b_sym, [])
        b_paired = pair_synchronized_close_contracts(
            b_contracts, opt_5m_map, und_5m_map, session_close_utc=close_utc,
        )
        sync_bench_metrics[b_sym] = compute_symbol_option_metrics_synchronized(
            b_paired, as_of_date=scored_day,
        )
        b_vwap_list = [
            (c, vwap_px_map[c.symbol][0], vwap_px_map[c.symbol][1])
            for c in b_contracts
            if c.symbol in vwap_px_map
        ]
        vwap_bench_metrics[b_sym] = compute_symbol_option_metrics(
            b_vwap_list, spot_close=spot_by_sym.get(b_sym, 0.0), as_of_date=scored_day,
        )

    sync_spy_term = _opt_finite_float(sync_bench_metrics.get("SPY", {}).get("iv_term_30_90"))
    sync_qqq_term = _opt_finite_float(sync_bench_metrics.get("QQQ", {}).get("iv_term_30_90"))
    sync_regime = resolve_opra_regime(sync_spy_term, sync_qqq_term)

    vwap_spy_term = _opt_finite_float(vwap_bench_metrics.get("SPY", {}).get("iv_term_30_90"))
    vwap_qqq_term = _opt_finite_float(vwap_bench_metrics.get("QQQ", {}).get("iv_term_30_90"))
    vwap_regime = resolve_opra_regime(vwap_spy_term, vwap_qqq_term)

    if use_proxy_for_primary and sync_regime == "unknown" and vwap_regime != "unknown":
        active_bench = vwap_bench_metrics
        opra_regime: OpraRegimeState = vwap_regime
        benchmark_obs_kind = "daily_vwap_proxy"
    else:
        active_bench = sync_bench_metrics
        opra_regime = sync_regime
        benchmark_obs_kind = (
            "synchronized_close"
            if (sync_spy_term is not None or sync_qqq_term is not None)
            else "unavailable"
        )

    spy_term = _opt_finite_float(active_bench.get("SPY", {}).get("iv_term_30_90"))
    qqq_term = _opt_finite_float(active_bench.get("QQQ", {}).get("iv_term_30_90"))
    opra_backwardation: bool | None = (
        True if opra_regime == "backwardation" else (False if opra_regime == "contango" else None)
    )

    # 2. Per-stock option metrics across candidate universe
    rx_by_sid: dict[str, float | None] = {}
    if "latest_catalyst_rx_1d" in enriched.columns:
        for _, r_row in enriched[["security_id", "latest_catalyst_rx_1d"]].iterrows():
            rv = r_row["latest_catalyst_rx_1d"]
            rx_by_sid[str(r_row["security_id"])] = None if rv is None or pd.isna(rv) else float(rv)

    metrics_by_sid: dict[str, dict[str, Any]] = {}
    vwap_by_sid: dict[str, dict[str, Any]] = {}
    obs_kind_by_sid: dict[str, str] = {}
    chain_rows: list[dict[str, Any]] = []
    for tk, sid in sid_by_tk.items():
        c_idx = col_of[sid]
        atr_v = float(atr_last[c_idx]) if np.isfinite(atr_last[c_idx]) else None
        rx_v = rx_by_sid.get(sid)
        tk_contracts = contracts_by_sym.get(tk, [])

        tk_paired = pair_synchronized_close_contracts(
            tk_contracts, opt_5m_map, und_5m_map, session_close_utc=close_utc,
        )
        m_sync = compute_symbol_option_metrics_synchronized(
            tk_paired,
            as_of_date=scored_day,
            atr_pct_14=atr_v,
            latest_catalyst_rx_1d=rx_v,
        )
        s_vwap_contracts = [
            (c, vwap_px_map[c.symbol][0], vwap_px_map[c.symbol][1])
            for c in tk_contracts
            if c.symbol in vwap_px_map
        ]
        m_vwap = compute_symbol_option_metrics(
            s_vwap_contracts,
            spot_close=spot_by_sym[tk],
            as_of_date=scored_day,
            atr_pct_14=atr_v,
            latest_catalyst_rx_1d=rx_v,
        )
        vwap_by_sid[sid] = m_vwap

        has_sync = int(m_sync["synchronized_contracts_used"]) > 0
        has_vwap = any(m_vwap.get(k) is not None for k in ("atm_straddle_pct", "iv_30d", "put_call_skew_25d"))
        if has_sync:
            m_primary = m_sync
            s_obs_kind = "synchronized_close"
        elif use_proxy_for_primary and has_vwap:
            m_primary = {
                **m_vwap,
                "synchronized_contracts_used": 0,
                "latest_option_bar_utc": None,
                "max_spot_skew_seconds": None,
                "max_close_staleness_seconds": None,
            }
            s_obs_kind = "daily_vwap_proxy"
        else:
            m_primary = m_sync
            s_obs_kind = "daily_vwap_proxy_only" if has_vwap else "unavailable"

        metrics_by_sid[sid] = m_primary
        obs_kind_by_sid[sid] = s_obs_kind
        chain_rows.append({
            "session_date": str(scored_day),
            "security_id": sid,
            "ticker": tk,
            "spot_close": spot_by_sym[tk],
            "option_observation_kind": s_obs_kind,
            **m_primary,
            "vwap_proxy_atm_straddle_pct": m_vwap.get("atm_straddle_pct"),
            "vwap_proxy_atm_iv_30d": m_vwap.get("iv_30d"),
            "vwap_proxy_put_call_skew_25d": m_vwap.get("put_call_skew_25d"),
        })

    # 3. Zero-fit `falling_knife_put_skew_veto` (`put_call_skew_25d >= 90th pct` AND `> 0.05`) on primary observations only
    valid_skews = np.asarray([
        float(m["put_call_skew_25d"])
        for m in metrics_by_sid.values()
        if m["put_call_skew_25d"] is not None and np.isfinite(m["put_call_skew_25d"])
    ], dtype=np.float64)
    skew_p90 = float(np.percentile(valid_skews, 90.0)) if valid_skews.size >= 5 else 0.10
    veto_cutoff = max(0.05, skew_p90)

    sids_all = enriched["security_id"].astype(str).tolist()
    enriched["atm_straddle_pct"] = [metrics_by_sid.get(s, {}).get("atm_straddle_pct") for s in sids_all]
    enriched["atm_iv_30d"] = [metrics_by_sid.get(s, {}).get("iv_30d") for s in sids_all]
    enriched["put_call_skew_25d"] = [metrics_by_sid.get(s, {}).get("put_call_skew_25d") for s in sids_all]
    enriched["atm_iv_to_atr_ratio"] = [metrics_by_sid.get(s, {}).get("atm_iv_to_atr_ratio") for s in sids_all]
    enriched["catalyst_shock_to_straddle"] = [metrics_by_sid.get(s, {}).get("catalyst_shock_to_straddle") for s in sids_all]
    enriched["option_observation_kind"] = [obs_kind_by_sid.get(s) for s in sids_all]
    enriched["vwap_proxy_atm_straddle_pct"] = [vwap_by_sid.get(s, {}).get("atm_straddle_pct") for s in sids_all]
    enriched["vwap_proxy_atm_iv_30d"] = [vwap_by_sid.get(s, {}).get("iv_30d") for s in sids_all]
    enriched["vwap_proxy_put_call_skew_25d"] = [vwap_by_sid.get(s, {}).get("put_call_skew_25d") for s in sids_all]

    veto_flags: list[bool] = []
    vetoed_rev: list[float | None] = []
    for s, rev_val in zip(sids_all, enriched["score_reversal_short"], strict=False):
        sk = metrics_by_sid.get(s, {}).get("put_call_skew_25d")
        is_veto = bool(sk is not None and np.isfinite(sk) and float(sk) >= veto_cutoff)
        veto_flags.append(is_veto)
        if is_veto or rev_val is None or pd.isna(rev_val):
            vetoed_rev.append(None)
        else:
            vetoed_rev.append(float(rev_val))
    enriched["falling_knife_put_skew_veto"] = veto_flags
    enriched["score_reversal_skew_vetoed"] = vetoed_rev

    # 4. Tri-state `R2` adaptive score (`None` when `opra_regime == "unknown"`, never defaulting to contango! `R7`)
    raw_arr = pd.to_numeric(enriched["score_price_equal_12"], errors="coerce").to_numpy(dtype=np.float64)
    bn_arr = pd.to_numeric(enriched["score_beta_neutral_price_12"], errors="coerce").to_numpy(dtype=np.float64)
    rv_arr = pd.to_numeric(enriched["score_reversal_skew_vetoed"], errors="coerce").to_numpy(dtype=np.float64)
    if opra_regime == "backwardation":
        blend_1d = np.where(np.isfinite(bn_arr), 0.70 * bn_arr + 0.30 * np.nan_to_num(rv_arr, nan=0.0), np.nan)
        r2_vals = centered_rank(blend_1d[None, :], np.isfinite(blend_1d)[None, :])[0]
        enriched["score_r2_opra_conditioned"] = [None if not np.isfinite(v) else float(v) for v in r2_vals]
    elif opra_regime == "contango":
        enriched["score_r2_opra_conditioned"] = [None if not np.isfinite(v) else float(v) for v in raw_arr]
    else:
        enriched["score_r2_opra_conditioned"] = [None] * len(enriched)

    # 5. Combine persisted `mid_small_beta_neutral_sec` XGBoost scores with live `OPRA` `25d` put-skew veto & `ATM straddle` shock!
    for h_tag in ("10d", "21d"):
        src_c = f"score_mid_small_bn_sec_xgb_{h_tag}"
        dst_c = f"score_mid_small_bn_sec_xgb_opra_{h_tag}"
        if src_c in enriched.columns:
            if opra_regime == "unknown":
                enriched[dst_c] = [None] * len(enriched)
            else:
                enriched[dst_c] = sec_catalyst_model.apply_opra_straddle_skew_overlay(
                    enriched, base_col=src_c, out_col=dst_c,
                )

    shock_gt_straddle = sum(
        1 for m in metrics_by_sid.values()
        if m["catalyst_shock_to_straddle"] is not None and float(m["catalyst_shock_to_straddle"]) > 1.0
    )
    summary = {
        "opra_enabled": True,
        "as_of_date": scored_day.isoformat(),
        "session_close_utc": close_utc.isoformat(),
        "option_observation_policy": opra_observation_policy,
        "benchmark_observation_kind": benchmark_obs_kind,
        "close_observation_window_minutes": CLOSE_OBSERVATION_WINDOW_MINUTES,
        "max_allowed_spot_skew_seconds": MAX_SPOT_SKEW_SECONDS,
        "max_allowed_close_staleness_seconds": MAX_CLOSE_STALENESS_SECONDS,
        "candidates_queried": len(sid_by_tk),
        "contracts_queried": len(all_contracts),
        "contracts_with_closing_5m_bars": len(opt_5m_map),
        "underlyings_with_closing_5m_bars": len(und_5m_map),
        "contracts_with_daily_vwap_proxy_bars": len(vwap_px_map),
        "contracts_with_session_bars": len(opt_5m_map),
        "stocks_with_synchronized_close": sum(1 for v in obs_kind_by_sid.values() if v == "synchronized_close"),
        "stocks_with_daily_vwap_proxy_only": sum(1 for v in obs_kind_by_sid.values() if v == "daily_vwap_proxy_only"),
        "stocks_with_atm_straddle": sum(1 for m in metrics_by_sid.values() if m["atm_straddle_pct"] is not None),
        "stocks_with_25d_skew": int(valid_skews.size),
        "falling_knife_veto_cutoff": veto_cutoff,
        "falling_knife_vetoed_count": int(sum(veto_flags)),
        "catalyst_shock_exceeds_straddle_count": shock_gt_straddle,
        "spy_iv_30d": active_bench.get("SPY", {}).get("iv_30d"),
        "spy_iv_90d": active_bench.get("SPY", {}).get("iv_90d"),
        "spy_iv_term_30_90": spy_term,
        "spy_latest_option_bar_utc": active_bench.get("SPY", {}).get("latest_option_bar_utc"),
        "spy_max_spot_skew_seconds": active_bench.get("SPY", {}).get("max_spot_skew_seconds"),
        "qqq_iv_30d": active_bench.get("QQQ", {}).get("iv_30d"),
        "qqq_iv_90d": active_bench.get("QQQ", {}).get("iv_90d"),
        "qqq_iv_term_30_90": qqq_term,
        "qqq_latest_option_bar_utc": active_bench.get("QQQ", {}).get("latest_option_bar_utc"),
        "qqq_max_spot_skew_seconds": active_bench.get("QQQ", {}).get("max_spot_skew_seconds"),
        "opra_regime": opra_regime,
        "opra_backwardation": opra_backwardation,
        "vwap_proxy_spy_iv_term_30_90": vwap_spy_term,
        "vwap_proxy_qqq_iv_term_30_90": vwap_qqq_term,
        "vwap_proxy_opra_regime": vwap_regime,
    }
    return enriched, pd.DataFrame(chain_rows), summary


__all__ = [
    "ALPACA_OPTIONS_BARS_URL",
    "ALPACA_STOCKS_BARS_URL",
    "CLOSE_OBSERVATION_WINDOW_MINUTES",
    "DEFAULT_RISK_FREE_RATE",
    "HISTORICAL_SEALED_END",
    "MAX_CLOSE_STALENESS_SECONDS",
    "MAX_SPOT_SKEW_SECONDS",
    "OccContract",
    "SynchronizedContractObservation",
    "bs_price_and_delta",
    "collect_live_opra_options_overlay",
    "compute_symbol_option_metrics",
    "compute_symbol_option_metrics_synchronized",
    "extract_snapshot_opra_regime",
    "pair_synchronized_close_contracts",
    "parse_occ_option_symbol",
    "resolve_opra_regime",
    "solve_iv_and_delta",
]
