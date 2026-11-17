"""OHLCV data loading, validation, synthetic generation and download."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]

_ALIASES = {
    "date": "date", "datetime": "date", "time": "date", "timestamp": "date",
    "日付": "date", "日時": "date", "年月日": "date",
    "open": "open", "始値": "open",
    "high": "high", "高値": "high",
    "low": "low", "安値": "low",
    "close": "close", "終値": "close",
    "adj close": "adj_close", "adj_close": "adj_close", "adjclose": "adj_close",
    "調整後終値": "adj_close",
    "volume": "volume", "vol": "volume", "出来高": "volume",
}


class DataError(ValueError):
    """Raised when price data is unusable."""


def validate_ohlcv(df: pd.DataFrame, repair_tolerance: float = 0.0) -> pd.DataFrame:
    """Check and normalise an OHLCV frame. Returns a cleaned copy.

    Rules: DatetimeIndex, sorted, unique, required columns present, positive
    finite prices, and low <= open/close <= high on every bar.

    repair_tolerance: some vendors publish bars whose high/low miss the
    open/close by a tiny amount. Bars off by at most this fraction are fixed
    by widening high/low; the count is stored in ``attrs["repaired_bars"]``.
    Larger violations always raise, because they indicate broken data.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("index must be a DatetimeIndex")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out[REQUIRED] = out[REQUIRED].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=REQUIRED)
    if len(out) < 2:
        raise DataError("need at least 2 valid bars")
    prices = out[REQUIRED].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DataError("prices must be positive and finite")
    top = out[["open", "close", "low"]].max(axis=1)
    bottom = out[["open", "close", "high"]].min(axis=1)
    gap = np.maximum(top / out["high"] - 1, 1 - bottom / out["low"]).clip(lower=0)
    bad = gap > 0
    repaired = 0
    if bad.any():
        if gap.max() > repair_tolerance:
            worst = gap.idxmax()
            raise DataError(
                f"{int(bad.sum())} bars violate low <= open/close <= high "
                f"(worst: {worst}, off by {gap.max():.4%})")
        out["high"] = np.maximum(out["high"], top)
        out["low"] = np.minimum(out["low"], bottom)
        repaired = int(bad.sum())
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out.index.name = "date"
    out = out[REQUIRED + ["volume"]].astype(float)
    out.attrs["repaired_bars"] = repaired
    return out


def load_csv(path: str | Path, adjust: bool = True,
             repair_tolerance: float = 0.0) -> pd.DataFrame:
    """Load an OHLCV CSV with flexible (English or Japanese) headers.

    If an adjusted-close column exists and ``adjust`` is true, OHLC is scaled
    by adj_close/close so that splits and dividends do not appear as gaps.
    """
    raw = pd.read_csv(path)
    raw.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                   for c in raw.columns]
    if "date" not in raw.columns:
        raise DataError("no date column found")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date")
    if adjust and "adj_close" in raw.columns:
        ratio = pd.to_numeric(raw["adj_close"], errors="coerce") / \
            pd.to_numeric(raw["close"], errors="coerce")
        for c in REQUIRED:
            raw[c] = pd.to_numeric(raw[c], errors="coerce") * ratio
    return validate_ohlcv(raw, repair_tolerance)


def synthetic_ohlcv(n: int = 1500, seed: int = 0, drift: float = 0.05,
                    vol: float = 0.2, start: str = "2015-01-01",
                    start_price: float = 1000.0) -> pd.DataFrame:
    """Random-walk (geometric Brownian motion) daily bars.

    For testing and demos only. It contains no exploitable structure, so a
    strategy that "wins" on it is winning by luck.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    prev = np.concatenate([[start_price], close[:-1]])
    open_ = prev * np.exp(0.2 * vol * np.sqrt(dt) * rng.standard_normal(n))
    span = np.abs(vol * np.sqrt(dt) * rng.standard_normal(n)) * 0.5
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    idx = pd.bdate_range(start=start, periods=n, name="date")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)
    return validate_ohlcv(df)


# --------------------------------------------------------------------------
# Download (requires network access to the data host)
# --------------------------------------------------------------------------

_UA = {"User-Agent": "Mozilla/5.0 (compatible; tradelab/0.1)"}


def _ts(d: str | None, default: datetime) -> int:
    dt = pd.Timestamp(d).to_pydatetime() if d else default
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def drop_forming_bar(df: pd.DataFrame, meta: dict, now: datetime) -> pd.DataFrame:
    """Remove today's daily bar while it is still forming.

    A bar dated today (UTC) is complete only if the instrument has a trading
    session and that session has ended. Markets that never close (crypto)
    keep forming the bar until the UTC day ends.
    """
    today = pd.Timestamp(now.date())
    end = ((meta.get("currentTradingPeriod") or {}).get("regular") or {}).get("end")
    never_closes = meta.get("instrumentType") == "CRYPTOCURRENCY"
    if not never_closes and end is not None and now.timestamp() >= end:
        return df
    return df[df.index < today]


def _fetch_yahoo(symbol: str, start: str | None, end: str | None,
                 interval: str, adjust: bool, timeout: float,
                 repair_tolerance: float) -> pd.DataFrame:
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": _ts(start, datetime(2000, 1, 1)),
        "period2": _ts(end, datetime.now(timezone.utc).replace(tzinfo=None)),
        "interval": interval, "events": "div,splits",
    }
    r = requests.get(url, params=params, headers=_UA, timeout=timeout)
    r.raise_for_status()
    chart = r.json().get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise DataError(f"yahoo returned no data for {symbol}: {chart.get('error')}")
    res = chart["result"][0]
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s")
    if interval.endswith(("d", "wk", "mo")):
        idx = idx.normalize()
    df = pd.DataFrame({k: q[k] for k in REQUIRED + ["volume"]}, index=idx)
    adj = res["indicators"].get("adjclose")
    if adjust and adj:
        ratio = pd.Series(adj[0]["adjclose"], index=idx, dtype=float) / df["close"]
        for c in REQUIRED:
            df[c] = df[c] * ratio
    if interval == "1d":
        df = drop_forming_bar(df, res.get("meta") or {}, datetime.now(timezone.utc))
    return validate_ohlcv(df, repair_tolerance)


def _fetch_stooq(symbol: str, start: str | None, end: str | None,
                 timeout: float, repair_tolerance: float) -> pd.DataFrame:
    import requests

    params = {"s": symbol.lower(), "i": "d"}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    r = requests.get("https://stooq.com/q/d/l/", params=params, headers=_UA,
                     timeout=timeout)
    r.raise_for_status()
    text = r.text.strip()
    if not text.lower().startswith("date"):
        raise DataError(f"stooq returned no data for {symbol}: {text[:80]!r}")
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    return validate_ohlcv(raw.set_index("date"), repair_tolerance)


def fetch_meta(symbol: str, timeout: float = 30.0) -> dict:
    """Name, currency and type of a Yahoo symbol (from the chart endpoint)."""
    import requests

    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"range": "5d", "interval": "1d"}, headers=_UA, timeout=timeout)
    r.raise_for_status()
    chart = r.json().get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise DataError(f"yahoo returned no data for {symbol}: {chart.get('error')}")
    m = chart["result"][0].get("meta") or {}
    return {"currency": m.get("currency") or "USD",
            "name": m.get("longName") or m.get("shortName") or "",
            "type": m.get("instrumentType") or "", "exchange": m.get("exchangeName") or ""}


def fetch_ohlcv(symbol: str, source: str = "yahoo", start: str | None = None,
                end: str | None = None, interval: str = "1d",
                adjust: bool = True, timeout: float = 30.0,
                repair_tolerance: float = 0.005) -> pd.DataFrame:
    """Download OHLCV bars.

    Bars whose high/low miss open/close by up to ``repair_tolerance`` (0.5%)
    are repaired, see ``validate_ohlcv``. Check ``attrs["repaired_bars"]``.

    source="yahoo": e.g. 7203.T (JP stock), AAPL (US stock), USDJPY=X (FX),
                    BTC-USD (crypto).
    source="stooq": e.g. 7203.jp, aapl.us, usdjpy, btcusd. Daily bars only.
    """
    if source == "yahoo":
        return _fetch_yahoo(symbol, start, end, interval, adjust, timeout,
                            repair_tolerance)
    if source == "stooq":
        if interval != "1d":
            raise ValueError("stooq source supports interval='1d' only")
        return _fetch_stooq(symbol, start, end, timeout, repair_tolerance)
    raise ValueError("source must be 'yahoo' or 'stooq'")
