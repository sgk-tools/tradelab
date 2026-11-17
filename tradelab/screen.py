"""CAN SLIM + Dow theory stock screener.

Every rule is a plain function of price/fundamental data so it can be tested
without network access. ``run_screen`` adds downloading, caching and ranking.

Criteria (thresholds live in ``ScreenConfig``):
  C  latest quarterly EPS growth vs the same quarter a year earlier
  A  annual EPS compound growth, rising every year, and return on equity
  N  price close to its 52-week high
  S  volume on up days vs down days over the last 50 bars (accumulation)
  L  relative strength rank within the screened universe
  I  institutional sponsorship: NOT evaluated (no data source available)
  M  market direction: the market index is in a Dow-theory uptrend and above
     its 200-day average
  DOW the stock itself is in a Dow-theory uptrend (higher highs, higher lows)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import fetch_ohlcv, load_csv
from .fundamentals import fetch_fundamentals_raw, parse_timeseries
from .strategies import dow_theory

CRITERIA = ["C", "A", "N", "S", "L", "DOW"]
MIN_BARS = 253

SP500_URL = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
             "main/data/constituents.csv")
# Major coins. Availability differs per exchange: confirm before trading.
CRYPTO = ["BTC", "ETH", "XRP", "LTC", "BCH", "XLM", "DOT", "LINK", "SOL",
          "DOGE", "ADA", "AVAX"]


@dataclass(frozen=True)
class ScreenConfig:
    c_min_eps_growth: float = 0.25    # C: quarterly EPS growth, year on year
    a_min_eps_cagr: float = 0.25      # A: annual EPS compound growth
    a_min_roe: float = 0.17           # A: return on equity
    n_max_below_high: float = 0.15    # N: max distance below the 52-week high
    s_min_updown_volume: float = 1.1  # S: up-day / down-day volume, 50 bars
    l_min_rs_rank: float = 80.0       # L: relative strength percentile (1-99)
    dow_swing: int = 5                # bars on each side of a swing point
    max_quarter_age_days: int = 200   # ignore stale quarterly results


# ---- individual measurements ------------------------------------------------

def quarterly_yoy_growth(series: pd.Series, asof: pd.Timestamp,
                         max_age_days: int = 200) -> float:
    """Growth of the latest quarter vs the quarter ~1 year earlier.

    NaN when the latest figure is stale, the year-ago quarter is missing, or
    the year-ago value is not positive (growth is undefined from a loss).
    """
    s = series.dropna()
    if s.empty or (asof - s.index[-1]).days > max_age_days:
        return np.nan
    latest_date = s.index[-1]
    gap = np.abs((latest_date - s.index[:-1]).days - 365)
    if len(gap) == 0 or gap.min() > 20:
        return np.nan
    prior = s.iloc[int(np.argmin(gap))]
    return (s.iloc[-1] - prior) / prior if prior > 0 else np.nan


def annual_eps_growth(series: pd.Series) -> tuple[float, bool]:
    """(compound annual growth over the available years, rose every year).

    Needs at least 3 annual figures; start and end must be positive.
    """
    s = series.dropna()
    if len(s) < 3 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan, False
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1
    return float(cagr), bool((s.diff().dropna() > 0).all())


def return_on_equity(net_income: pd.Series, equity: pd.Series) -> float:
    ni, eq = net_income.dropna(), equity.dropna()
    if ni.empty or eq.empty or ni.index[-1] != eq.index[-1] or eq.iloc[-1] <= 0:
        return np.nan
    return float(ni.iloc[-1] / eq.iloc[-1])


def below_52w_high(data: pd.DataFrame) -> float:
    return float(1 - data["close"].iloc[-1] / data["high"].iloc[-252:].max())


def updown_volume_ratio(data: pd.DataFrame, bars: int = 50) -> float:
    chg = data["close"].diff().iloc[-bars:]
    vol = data["volume"].iloc[-bars:]
    down = vol[chg < 0].sum()
    return float(vol[chg > 0].sum() / down) if down > 0 else np.nan


def rs_raw(data: pd.DataFrame) -> float:
    """Weighted 12-month performance; the latest quarter counts double."""
    c = data["close"]
    r = [c.iloc[-1] / c.iloc[-1 - n] - 1 for n in (63, 126, 189, 252)]
    return float(0.4 * r[0] + 0.2 * r[1] + 0.2 * r[2] + 0.2 * r[3])


def trend_state(data: pd.DataFrame, swing: int = 5) -> tuple[int, int]:
    """(current Dow trend: 1/0/-1, bars since it last changed)."""
    t = dow_theory(data, swing=swing, allow_short=True).to_numpy()
    changed = np.flatnonzero(t != t[-1])
    since = len(t) - 1 - (changed[-1] if len(changed) else -1)
    return int(t[-1]), int(since)


def market_status(index_data: pd.DataFrame, swing: int = 5) -> dict:
    """M criterion on a market index."""
    trend, since = trend_state(index_data, swing)
    close = index_data["close"]
    sma200 = close.rolling(200).mean().iloc[-1]
    above = bool(close.iloc[-1] > sma200)
    return {"trend": trend, "bars_in_trend": since, "above_sma200": above,
            "close": float(close.iloc[-1]), "sma200": float(sma200),
            "pass": trend == 1 and above}


def measure(prices: pd.DataFrame, fundamentals: dict | None,
            cfg: ScreenConfig) -> dict:
    """Raw measurements for one symbol (no pass/fail decisions yet)."""
    trend, since = trend_state(prices, cfg.dow_swing)
    row = {
        "close": float(prices["close"].iloc[-1]),
        "below_high": below_52w_high(prices),
        "updown_volume": updown_volume_ratio(prices),
        "rs_raw": rs_raw(prices),
        "dow_trend": trend, "dow_bars": since,
        "eps_growth_q": np.nan, "revenue_growth_q": np.nan,
        "eps_cagr": np.nan, "eps_rising": False, "roe": np.nan,
    }
    if fundamentals:
        asof = prices.index[-1]
        cagr, rising = annual_eps_growth(fundamentals["annual_eps"])
        row.update(
            eps_growth_q=quarterly_yoy_growth(
                fundamentals["quarterly_eps"], asof, cfg.max_quarter_age_days),
            revenue_growth_q=quarterly_yoy_growth(
                fundamentals["quarterly_revenue"], asof, cfg.max_quarter_age_days),
            eps_cagr=cagr, eps_rising=rising,
            roe=return_on_equity(fundamentals["annual_net_income"],
                                 fundamentals["annual_equity"]))
    return row


def score(table: pd.DataFrame, cfg: ScreenConfig, market_pass: bool,
          use_fundamentals: bool = True,
          rs_reference: pd.Series | None = None) -> pd.DataFrame:
    """Add RS rank, pass/fail columns, score and candidate flag; sort.

    ``rs_reference``: raw RS values of another universe to rank against
    (used for a short watch list, whose own ranking would be meaningless).
    """
    t = table.copy()
    if rs_reference is not None and len(rs_reference.dropna()):
        ref = np.sort(rs_reference.dropna().to_numpy())
        pctl = np.searchsorted(ref, t["rs_raw"].to_numpy(), side="right") / len(ref)
        t["rs_rank"] = np.where(t["rs_raw"].notna(), np.clip(pctl * 98 + 1, 1, 99).round(), np.nan)
    else:
        t["rs_rank"] = (t["rs_raw"].rank(pct=True) * 98 + 1).round()
    t["C"] = t["eps_growth_q"] >= cfg.c_min_eps_growth
    t["A"] = (t["eps_cagr"] >= cfg.a_min_eps_cagr) & t["eps_rising"].astype(bool) \
        & (t["roe"] >= cfg.a_min_roe)
    t["N"] = t["below_high"] <= cfg.n_max_below_high
    t["S"] = t["updown_volume"] >= cfg.s_min_updown_volume
    t["L"] = t["rs_rank"] >= cfg.l_min_rs_rank
    t["DOW"] = t["dow_trend"] == 1
    used = CRITERIA if use_fundamentals else ["N", "S", "L", "DOW"]
    t["score"] = t[used].sum(axis=1).astype(int)
    t["max_score"] = len(used)
    t["candidate"] = (t["score"] == len(used)) & market_pass
    return t.sort_values(["score", "rs_rank"], ascending=False, kind="stable")


# ---- universe -----------------------------------------------------------------

def load_universe(name: str | None = None, file: str | None = None) -> pd.DataFrame:
    """Columns: symbol (Yahoo format), name, sector."""
    if file:
        syms = [ln.split("#")[0].strip().upper()
                for ln in Path(file).read_text(encoding="utf-8").splitlines()]
        syms = list(dict.fromkeys(s for s in syms if s))
        if not syms:
            raise ValueError(f"no symbols in {file}")
        return pd.DataFrame({"symbol": syms, "name": "", "sector": ""})
    if name == "crypto":
        return pd.DataFrame({"symbol": [f"{c}-USD" for c in CRYPTO],
                             "name": CRYPTO, "sector": "crypto"})
    if name == "sp500":
        import io

        import requests
        r = requests.get(SP500_URL, timeout=30)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        return pd.DataFrame({
            "symbol": raw["Symbol"].str.strip().str.replace(".", "-", regex=False),
            "name": raw["Security"], "sector": raw["GICS Sector"]})
    raise ValueError("universe must be 'sp500' or 'crypto', or pass a file")


WATCH_MAX = 50
_SYMBOL = __import__("re").compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,19}$")


def parse_watch(text: str, limit: int = WATCH_MAX) -> list[str]:
    """Symbols from a watch-list file: one per line, '#' comments allowed.

    Upper-cased, de-duplicated, invalid entries dropped, at most ``limit``.
    """
    out: list[str] = []
    for line in text.splitlines():
        sym = line.split("#")[0].strip().upper()
        if sym and _SYMBOL.match(sym) and sym not in out:
            out.append(sym)
    return out[:limit]


def _cached_meta(symbol: str, cache: Path) -> dict:
    from .data import fetch_meta
    path = cache / "meta" / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    meta = _retry(lambda: fetch_meta(symbol))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta))
    return meta


# ---- download with cache and retry -----------------------------------------

def _retry(fn: Callable, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if i == attempts - 1 or status in (400, 401, 403, 404):
                raise
            time.sleep(1.5 * 2 ** i)


def _cached_prices(symbol: str, cache: Path, start: str) -> pd.DataFrame:
    path = cache / "prices" / f"{symbol}.csv"
    if path.exists():
        return load_csv(path)
    data = _retry(lambda: fetch_ohlcv(symbol, start=start))
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)
    return data


def _cached_fundamentals(symbol: str, cache: Path) -> dict:
    path = cache / "fundamentals" / f"{symbol}.json"
    if path.exists():
        return parse_timeseries(json.loads(path.read_text()))
    raw = _retry(lambda: fetch_fundamentals_raw(symbol))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    return parse_timeseries(raw)


@dataclass
class ScreenResult:
    table: pd.DataFrame       # one row per screened symbol, best first
    market: dict              # index label -> market_status dict
    market_pass: bool
    failures: pd.DataFrame    # symbols that could not be evaluated, with reason
    config: ScreenConfig
    asof: pd.Timestamp
    universe: str
    use_fundamentals: bool


def run_screen(universe: pd.DataFrame, cfg: ScreenConfig | None = None,
               market_symbols: dict[str, str] | None = None,
               use_fundamentals: bool = True, cache_dir: str | Path = "data/cache",
               workers: int = 8, universe_name: str = "custom",
               progress: Callable[[int, int], None] | None = None,
               price_loader: Callable | None = None,
               fundamentals_loader: Callable | None = None,
               history_days: int = 800,
               rs_reference: pd.Series | None = None) -> ScreenResult:
    """Download (or read from today's cache) and screen every symbol.

    The first entry of ``market_symbols`` decides the M criterion; the others
    are reported for confirmation. Loaders can be injected for testing.
    """
    cfg = cfg or ScreenConfig()
    market_symbols = market_symbols or {"S&P 500": "^GSPC"}
    today = datetime.now(timezone.utc).date()
    cache = Path(cache_dir) / today.isoformat()
    start = (today - timedelta(days=history_days)).isoformat()
    get_prices = price_loader or (lambda s: _cached_prices(s, cache, start))
    get_fund = fundamentals_loader or (lambda s: _cached_fundamentals(s, cache))

    market = {label: market_status(get_prices(sym), cfg.dow_swing)
              for label, sym in market_symbols.items()}
    market_pass = next(iter(market.values()))["pass"]

    def work(symbol: str):
        try:
            prices = get_prices(symbol)
            if len(prices) < MIN_BARS:
                return symbol, None, f"価格データ不足 ({len(prices)}本)", prices.index[-1]
            fund = get_fund(symbol) if use_fundamentals else None
            return symbol, measure(prices, fund, cfg), None, prices.index[-1]
        except Exception as exc:
            return symbol, None, f"{type(exc).__name__}: {exc}"[:120], None

    symbols = list(universe["symbol"])
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in enumerate(pool.map(work, symbols), 1):
            results.append(res)
            if progress:
                progress(i, len(symbols))

    last_dates = [d for _, row, _, d in results if row is not None]
    asof = max(last_dates) if last_dates else pd.Timestamp(today)
    rows, failures = [], []
    for symbol, row, error, last in results:
        if row is not None and (asof - last).days > 7:
            row, error = None, f"価格が古い (最終 {last.date()})"
        if row is None:
            failures.append({"symbol": symbol, "reason": error})
        else:
            rows.append({"symbol": symbol, **row})
    if not rows:
        raise ValueError("no symbol could be evaluated: "
                         + "; ".join(f"{f['symbol']}: {f['reason']}" for f in failures[:3]))
    table = pd.DataFrame(rows).merge(universe, on="symbol", how="left")
    table = score(table, cfg, market_pass, use_fundamentals,
                  rs_reference).reset_index(drop=True)
    return ScreenResult(table, market, market_pass,
                        pd.DataFrame(failures, columns=["symbol", "reason"]),
                        cfg, asof, universe_name, use_fundamentals)
