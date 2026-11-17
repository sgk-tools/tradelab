"""Build the mobile ranking app: one self-contained HTML file.

The page embeds the day's screening result (ranking, criteria, price history
and earnings history for the top symbols) so it works on any static host or
as a published page with no server.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .screen import ScreenResult
from .swing import QUALITY, QUALITY_TECH, SIGNALS, SwingConfig, add_swing

TEMPLATE = Path(__file__).with_name("app_template.html")
JST = timezone(timedelta(hours=9))
CHART_POINTS = 126
OPEN_BARS = 60          # daily opens kept for demo-trade fills

_STANDALONE_HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="スイング銘柄">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);
padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}
img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
"""


def _num(value, digits: int = 4):
    """JSON-safe number: None for NaN/inf."""
    if value is None:
        return None
    v = float(value)
    return round(v, digits) if np.isfinite(v) else None


def _chart(prices: pd.DataFrame, fundamentals: dict | None) -> dict:
    tail = prices.iloc[-252:]
    step = max(1, len(tail) // CHART_POINTS)
    pick = tail.iloc[::-1].iloc[::step].iloc[::-1]       # always keeps the last bar
    out = {"d": [d.strftime("%Y-%m-%d") for d in pick.index],
           "c": [_num(v, 4) for v in pick["close"]],
           "hi": _num(tail["high"].max(), 4)}
    if fundamentals:
        out["q"] = [[d.strftime("%Y-%m"), _num(v, 3)]
                    for d, v in fundamentals["quarterly_eps"].dropna().iloc[-5:].items()]
        out["a"] = [[d.strftime("%Y"), _num(v, 3)]
                    for d, v in fundamentals["annual_eps"].dropna().iloc[-4:].items()]
    return out


def _mark(r, c: str) -> str:
    """'1' pass, '0' fail, '-' not measurable (earnings data missing)."""
    missing = {"C": ["eps_growth_q"], "A": ["eps_cagr", "roe"]}.get(c, [])
    if not bool(r[c]) and any(pd.isna(r.get(m)) for m in missing):
        return "-"
    return "1" if bool(r[c]) else "0"


def _list(result: ScreenResult, label: str, prev_ranks: dict[str, int],
          currency: dict[str, str] | None = None) -> dict:
    t = result.table
    crit = QUALITY if result.use_fundamentals else QUALITY_TECH
    rows = []
    for rank, (_, r) in enumerate(t.iterrows(), 1):
        rows.append({
            "s": r["symbol"], "n": str(r.get("name") or ""), "sec": str(r.get("sector") or ""),
            "sc": int(r["swing_score"]), "cand": bool(r["swing_candidate"]),
            "p": "".join(_mark(r, c) for c in crit),
            "g": "".join("1" if bool(r[c]) else "0" for c in SIGNALS),
            "st": _num(r.get("stop"), 4), "tg": _num(r.get("target"), 4),
            "rp": _num(r.get("risk_pct")), "at": _num(r.get("atr"), 4),
            "r2": _num(r.get("rsi2"), 1),
            "eg": _num(r["eps_growth_q"]), "rg": _num(r["revenue_growth_q"]),
            "ca": _num(r["eps_cagr"]), "roe": _num(r["roe"]),
            "bh": _num(r["below_high"]), "uv": _num(r["updown_volume"], 2),
            "rs": _num(r["rs_rank"], 0), "dt": int(r["dow_trend"]), "db": int(r["dow_bars"]),
            "cl": _num(r["close"], 4), "prev": prev_ranks.get(r["symbol"]),
            "cu": (currency or {}).get(r["symbol"], "USD"),
        })
    return {
        "label": label, "universe": result.universe, "asof": result.asof.strftime("%Y-%m-%d"),
        "marketPass": bool(result.market_pass), "crit": crit, "sigs": SIGNALS,
        "max": len(crit) + 1, "rows": rows,
        "failed": int(len(result.failures)),
        "fails": [{"s": f["symbol"], "why": str(f["reason"])[:80]}
                  for f in result.failures.to_dict("records")][:60],
        "markets": [{"label": k, "pass": bool(m["pass"]), "trend": int(m["trend"]),
                     "bars": int(m["bars_in_trend"]), "above": bool(m["above_sma200"])}
                    for k, m in result.market.items()],
    }


def _sig(value, digits: int = 6):
    """JSON-safe number rounded to significant digits (keeps the page small)."""
    v = _num(value, 12)
    return None if v is None else float(f"{v:.{digits}g}")


def _opens(symbols, price_loader: Callable[[str], pd.DataFrame],
            bars: int = OPEN_BARS) -> dict:
    """Recent daily open/high/low on one shared date axis.

    {"d": dates, "o": {sym: [...]}, "h": {...}, "l": {...}}. The page fills
    a demo order at the first open after it was placed and triggers its
    stop / target sells from the highs and lows.
    """
    series, prev = {}, {}
    for sym in symbols:
        try:
            data = price_loader(sym)
            series[sym] = data[["open", "high", "low"]].iloc[-bars:]
            prev[sym] = _sig(data["close"].iloc[-2]) if len(data) > 1 else None
        except Exception:
            continue
    out = {"d": [], "o": {}, "h": {}, "l": {}, "pc": {}}
    if not series:
        return out
    dates = sorted(set().union(*(x.index for x in series.values())))[-bars:]
    axis = pd.DatetimeIndex(dates)
    out["d"] = [d.strftime("%Y-%m-%d") for d in axis]
    for sym, x in series.items():
        x = x.reindex(axis)
        for col, key in (("open", "o"), ("high", "h"), ("low", "l")):
            out[key][sym] = [_sig(v) for v in x[col]]
    out["pc"] = prev
    return out


def build_app_data(results: dict[str, tuple[str, ScreenResult]],
                   price_loader: Callable[[str], pd.DataFrame],
                   fundamentals_loader: Callable[[str], dict],
                   previous: dict | None = None, charts_per_list: int = 60,
                   now: datetime | None = None, stats: dict | None = None,
                   fx: dict | None = None, swing_cfg: SwingConfig | None = None,
                   currencies: dict[str, dict[str, str]] | None = None) -> dict:
    """``results`` maps a list key to (display label, ScreenResult).

    ``previous`` is the data of the last build; when its price date differs,
    it supplies each symbol's previous rank for the "change" marker.
    """
    swing_cfg = swing_cfg or SwingConfig()
    lists, charts, opens = {}, {}, {}
    for key, (label, res) in results.items():
        res = replace(res, table=add_swing(res.table, price_loader, res.market_pass,
                                           res.use_fundamentals, swing_cfg))
        prev_ranks: dict[str, int] = {}
        old = (previous or {}).get("lists", {}).get(key)
        if old and old.get("asof") != res.asof.strftime("%Y-%m-%d"):
            prev_ranks = {r["s"]: i for i, r in enumerate(old["rows"], 1)}
        elif old:                                  # same day rebuilt: keep markers
            prev_ranks = {r["s"]: r["prev"] for r in old["rows"] if r.get("prev")}
        lists[key] = _list(res, label, prev_ranks, (currencies or {}).get(key))
        lists[key]["hasPrev"] = bool(prev_ranks)
        opens[key] = _opens(res.table["symbol"], price_loader)
        for row in lists[key]["rows"]:
            row["pc"] = opens[key]["pc"].get(row["s"])
        t = res.table
        for sym in t[t["swing_candidate"] | (t.index < charts_per_list)]["symbol"]:
            if "q" in charts.get(sym, {}):
                continue                           # keep the chart that has earnings
            try:
                fund = fundamentals_loader(sym) if res.use_fundamentals else None
                charts[sym] = _chart(price_loader(sym), fund)
            except Exception:
                continue                           # row still listed, without a chart
    first = next(iter(results.values()))[1]
    return {
        "generated": (now or datetime.now(JST)).astimezone(JST).strftime("%Y-%m-%d %H:%M"),
        "config": {**asdict(first.config), **asdict(swing_cfg)},
        "lists": lists, "charts": charts, "stats": stats or {}, "fx": fx,
        "opens": opens,
    }


def render_app(data: dict, out_dir: str | Path, split_bars: bool = True,
               state_dir: str | Path | None = None) -> dict[str, Path]:
    """Write page.html (page fragment), index.html (standalone) and data.json.

    With ``split_bars`` the daily bars used for demo fills go to
    bars_<list>.json next to the page (loaded by the page after it opens),
    keeping the page itself small. data.json is the small state the next
    build needs (each list's date and ranking); it goes to ``state_dir``
    (default: ``out_dir``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    if split_bars and data.get("opens"):
        for key, bars in data["opens"].items():
            paths[f"bars_{key}"] = out / f"bars_{key}.json"
            paths[f"bars_{key}"].write_text(json.dumps(bars, separators=(",", ":")), encoding="utf-8")
        page_data = {**data, "opens": {}, "barsFiles": {k: f"bars_{k}.json" for k in data["opens"]}}
    else:
        page_data = data
    payload = json.dumps(page_data, ensure_ascii=False, separators=(",", ":"))
    state = {"generated": data.get("generated"),
             "lists": {k: {"asof": L["asof"], "rows": [{"s": r["s"], "prev": r.get("prev")} for r in L["rows"]]}
                       for k, L in data.get("lists", {}).items()}}
    full = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    state_path = Path(state_dir or out)
    state_path.mkdir(parents=True, exist_ok=True)
    fragment = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", payload.replace("</", "<\\/"))
    paths.update({"page": out / "page.html", "index": out / "index.html",
                  "data": state_path / "data.json"})
    paths["page"].write_text(fragment, encoding="utf-8")
    paths["index"].write_text(_STANDALONE_HEAD + fragment + "\n</body></html>\n",
                              encoding="utf-8")
    paths["data"].write_text(full, encoding="utf-8")
    return paths


def load_previous(out_dir: str | Path) -> dict | None:
    path = Path(out_dir) / "data.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
