"""Company fundamentals (EPS, revenue, net income, equity) from Yahoo Finance."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .data import _UA

FIELDS = {
    "quarterly_eps": "quarterlyDilutedEPS",
    "annual_eps": "annualDilutedEPS",
    "quarterly_revenue": "quarterlyTotalRevenue",
    "annual_net_income": "annualNetIncome",
    "annual_equity": "annualStockholdersEquity",
}


def parse_timeseries(payload: dict) -> dict[str, pd.Series]:
    """Convert Yahoo's fundamentals-timeseries JSON into date-indexed Series.

    Missing fields yield empty Series, never an exception: many companies
    lack some items and the screener treats those as "cannot evaluate".
    """
    found: dict[str, pd.Series] = {}
    for item in (payload.get("timeseries") or {}).get("result") or []:
        kind = (item.get("meta", {}).get("type") or [None])[0]
        points = {}
        for v in item.get(kind) or []:
            raw = ((v or {}).get("reportedValue") or {}).get("raw")
            if raw is not None and v.get("asOfDate"):
                points[pd.Timestamp(v["asOfDate"])] = float(raw)
        found[kind] = pd.Series(points, dtype=float).sort_index()
    return {name: found.get(kind, pd.Series(dtype=float))
            for name, kind in FIELDS.items()}


def fetch_fundamentals_raw(symbol: str, timeout: float = 30.0) -> dict:
    import requests

    url = ("https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/"
           f"finance/timeseries/{symbol}")
    now = int(datetime.now(timezone.utc).timestamp())
    params = {"type": ",".join(FIELDS.values()),
              "period1": now - 6 * 366 * 86400, "period2": now}
    r = requests.get(url, params=params, headers=_UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_fundamentals(symbol: str, timeout: float = 30.0) -> dict[str, pd.Series]:
    return parse_timeseries(fetch_fundamentals_raw(symbol, timeout))
