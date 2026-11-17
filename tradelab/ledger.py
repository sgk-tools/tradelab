"""Automatic paper-trading record of the ranking (forward test).

Every daily build adds the top-ranked symbols that show a timing signal as
paper trades, then re-evaluates every recorded trade from price data with the
same plan the app shows: enter at the next open, stop stop_pct below,
target target_pct above, exit after max_hold bars.

Only the signal (symbol, date, risk fraction, context) is stored. Entry,
exit and result are recomputed from prices on every build, so the record
cannot drift from the data and a missed day does not lose exits.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .swing import SwingConfig, simulate_trade

PER_DAY = 10          # trades added per list per signal date


def add_signals(ledger: dict, key: str, rows: list[dict], asof: str,
                market_pass: bool, per_day: int = PER_DAY) -> int:
    """Record today's top rows with a signal. Returns how many were added.

    Each price date is recorded once per list (a same-day rebuild adds
    nothing), and a symbol is skipped while an earlier record of it in this
    list is still pending or open.
    """
    trades = ledger.setdefault("trades", [])
    ledger.setdefault("started", {}).setdefault(key, asof)
    done = ledger.setdefault("recorded", {}).setdefault(key, [])
    if asof in done:
        return 0                             # this price date was already recorded
    done.append(asof)
    busy = {t["s"] for t in trades if t["k"] == key and t.get("st") in (None, "pending", "open")}
    seen = {(t["k"], t["s"], t["d"]) for t in trades}
    added = 0
    for rank, r in enumerate(rows, 1):
        if added >= per_day:
            break
        if "1" not in r["g"] or r.get("rp") is None:
            continue
        if r["s"] in busy or (key, r["s"], asof) in seen:
            continue
        trades.append({"k": key, "s": r["s"], "d": asof, "rk": rank, "sc": r["sc"],
                       "g": r["g"], "cand": bool(r["cand"]), "m": bool(market_pass),
                       "rp": r["rp"], "st": "pending"})
        busy.add(r["s"])
        added += 1
    return added


def evaluate(ledger: dict, price_loader: Callable[[str], pd.DataFrame],
             cfg: SwingConfig | None = None, cost: dict[str, float] | None = None) -> dict:
    """Recompute entry/exit/R of every trade in place; return the ledger."""
    cfg = cfg or SwingConfig()
    cost = cost or {}
    cache: dict[str, pd.DataFrame | None] = {}
    for t in ledger.get("trades", []):
        if t["s"] not in cache:
            try:
                cache[t["s"]] = price_loader(t["s"])
            except Exception:
                cache[t["s"]] = None
        data = cache[t["s"]]
        if data is None:
            continue                         # keep the last known state
        idx = data.index.searchsorted(pd.Timestamp(t["d"]), side="right") - 1
        if idx < 0 or data.index[idx].strftime("%Y-%m-%d") != t["d"]:
            continue                         # signal bar not in this data
        o, h, lo, c = (data[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        res = simulate_trade(o, h, lo, c, idx, float(t["rp"]), cfg)
        if res is None:
            t.update(st="pending")
            continue
        entry, stop, target, xi, xp, why = res
        fee = cost.get(t["k"], cfg.cost_stock) * entry
        t.update(ed=data.index[idx + 1].strftime("%Y-%m-%d"), ep=_r(entry),
                 sp=_r(stop), tp=_r(target))
        if xi is None:
            t.update(st="open", xd=None, xp=None, why=None,
                     r=_r((c[-1] - entry - fee) / (entry - stop), 3),
                     last=_r(c[-1]), bars=int(len(c) - 1 - idx))
        else:
            t.update(st="closed", xd=data.index[xi].strftime("%Y-%m-%d"), xp=_r(xp),
                     why=why, r=_r((xp - entry - fee) / (entry - stop), 3),
                     bars=int(xi - idx))
    return ledger


def summary(ledger: dict, key: str) -> dict:
    trades = [t for t in ledger.get("trades", []) if t["k"] == key]
    closed = [t for t in trades if t.get("st") == "closed" and t.get("r") is not None]

    def stats(ts):
        r = np.array([t["r"] for t in ts], float)
        return {"n": int(len(r)), "win": _r(float((r > 0).mean()), 3) if len(r) else None,
                "avg": _r(float(r.mean()), 3) if len(r) else None,
                "sum": _r(float(r.sum()), 3) if len(r) else 0.0}
    return {"started": ledger.get("started", {}).get(key),
            "closed": stats(closed), "closedM": stats([t for t in closed if t["m"]]),
            "open": sum(t.get("st") == "open" for t in trades),
            "pending": sum(t.get("st") == "pending" for t in trades)}


def _r(v, digits: int = 4):
    if v is None:
        return None
    v = float(v)
    return round(v, digits) if np.isfinite(v) else None


def for_page(ledger: dict, key: str, recent: int = 40) -> dict:
    """Summary plus open trades and the most recent closed ones for one list."""
    trades = [t for t in ledger.get("trades", []) if t["k"] == key]
    live = [t for t in trades if t.get("st") in ("open", "pending")]
    done = sorted((t for t in trades if t.get("st") == "closed"),
                  key=lambda t: (t["xd"], t["s"]), reverse=True)[:recent]
    return {"summary": summary(ledger, key), "live": live, "done": done}


def load(path) -> dict:
    import json
    from pathlib import Path
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "started": {}, "trades": []}


def save(ledger: dict, path) -> None:
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps(ledger, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False), encoding="utf-8")
