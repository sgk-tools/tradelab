"""Swing-trading layer on top of the CAN SLIM + Dow screen.

Adds, for every screened symbol, price-only rules used by swing traders:

  TT   Trend Template (M. Minervini): price above rising long averages, the
       averages stacked 50 > 150 > 200, at least 30% above the 52-week low and
       within 25% of the 52-week high. (His relative-strength rule is the
       existing L criterion.)
  BO   Breakout: close above the highest high of the previous 20 bars on
       volume at least 1.5x the 50-bar average.
  PP   Pocket pivot (Morales & Kacher): an up day whose volume exceeds every
       down-day volume of the previous 10 bars, closing above the 50-bar average.
  PB   Pullback (L. Connors): RSI(2) below 10 while above the 200-bar average.
  VCP  Volatility contraction (simplified): the last 10 bars' range is tight
       relative to the 50-bar ATR and recent ATR has shrunk, near the 52-week high.

TT is a "quality" condition like C/A/N/S/L/DOW. BO/PP/PB/VCP are "timing"
signals: whether today is a typical entry day for a swing trade.

Every symbol also gets a trade plan following W. O'Neil's rules: stop
stop_pct (7%) below, target target_pct (20%) above, exit after max_hold bars
at the latest. ``signal_stats`` replays the same plan over history (enter
next open, exit at stop, target or after max_hold bars) so each signal's
record can be compared with entering on any day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .strategies import rsi

QUALITY = ["C", "A", "N", "S", "L", "DOW", "TT"]
QUALITY_TECH = ["N", "S", "L", "DOW", "TT"]
SIGNALS = ["BO", "PP", "PB", "VCP"]


@dataclass(frozen=True)
class SwingConfig:
    bo_lookback: int = 20         # BO: bars of prior highs to break
    bo_volume: float = 1.5        # BO: volume vs 50-bar average
    pp_lookback: int = 10         # PP: down-day volumes to beat
    pb_rsi: float = 10.0          # PB: RSI(2) below this
    vcp_range_atr: float = 2.5    # VCP: 10-bar range <= this x ATR(50)
    vcp_atr_ratio: float = 0.75   # VCP: ATR(10) / ATR(50) <= this
    vcp_near_high: float = 0.15   # VCP: within this of the 52-week high
    stop_pct: float = 0.07        # plan: stop this far below the entry (O'Neil 7%)
    target_pct: float = 0.20      # plan: target this far above (O'Neil 20%)
    max_hold: int = 60            # plan: exit after this many bars at the latest
    cost_stock: float = 0.002     # assumed round-trip cost (fraction of price)
    cost_crypto: float = 0.003

    @property
    def target_r(self) -> float:
        """Target as a multiple of the risk (20% / 7% = 2.86 by default)."""
        return self.target_pct / self.stop_pct


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = data["close"].shift(1)
    tr = pd.concat([data["high"] - data["low"], (data["high"] - prev).abs(),
                    (data["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def signal_frame(data: pd.DataFrame, cfg: SwingConfig | None = None) -> pd.DataFrame:
    """Boolean TT/BO/PP/PB/VCP for every bar, using only data up to that bar."""
    cfg = cfg or SwingConfig()
    c, h, lo = data["close"], data["high"], data["low"]
    v = data["volume"] if "volume" in data else pd.Series(np.nan, index=data.index)
    s50, s150, s200 = (c.rolling(n).mean() for n in (50, 150, 200))
    hi52, lo52 = h.rolling(252).max(), lo.rolling(252).min()
    tt = ((c > s150) & (c > s200) & (s150 > s200) & (s200 > s200.shift(21))
          & (s50 > s150) & (c > s50) & (c >= 1.3 * lo52) & (c >= 0.75 * hi52))

    avg_vol = v.rolling(50).mean().shift(1)
    bo = (c > h.rolling(cfg.bo_lookback).max().shift(1)) & (v >= cfg.bo_volume * avg_vol)

    down_vol = v.where(c.diff() < 0, 0.0)
    pp = (c.diff() > 0) & (v > down_vol.rolling(cfg.pp_lookback).max().shift(1)) \
        & (c > s50) & (v > 0)

    pb = (rsi(c, 2) < cfg.pb_rsi) & (c > s200)

    a50 = atr(data, 50)
    rng10 = h.rolling(10).max() - lo.rolling(10).min()
    vcp = (rng10 <= cfg.vcp_range_atr * a50) & (atr(data, 10) <= cfg.vcp_atr_ratio * a50) \
        & (c >= (1 - cfg.vcp_near_high) * hi52)

    out = pd.DataFrame({"TT": tt, "BO": bo, "PP": pp, "PB": pb, "VCP": vcp})
    return out.fillna(False).astype(bool)


def trade_plan(data: pd.DataFrame, cfg: SwingConfig | None = None) -> dict:
    cfg = cfg or SwingConfig()
    close = float(data["close"].iloc[-1])
    a = float(atr(data, 14).iloc[-1])            # shown for reference only
    if not np.isfinite(close) or close <= 0:
        return {"atr": a, "stop": np.nan, "target": np.nan, "risk_pct": np.nan}
    return {"atr": a, "stop": close * (1 - cfg.stop_pct),
            "target": close * (1 + cfg.target_pct), "risk_pct": cfg.stop_pct}


def add_swing(table: pd.DataFrame, price_loader: Callable[[str], pd.DataFrame],
              market_pass: bool, use_fundamentals: bool,
              cfg: SwingConfig | None = None) -> pd.DataFrame:
    """Add TT, signals, trade plan, swing score and candidate flag; re-sort.

    swing_score = quality conditions passed + 1 if any timing signal fired.
    swing_candidate = every quality condition, at least one signal, market OK.
    """
    cfg = cfg or SwingConfig()
    rows = []
    for sym in table["symbol"]:
        try:
            data = price_loader(sym)
            last = signal_frame(data, cfg).iloc[-1]
            rows.append({**last.to_dict(), **trade_plan(data, cfg),
                         "rsi2": float(rsi(data["close"], 2).iloc[-1])})
        except Exception:
            rows.append({k: False for k in ["TT"] + SIGNALS})
    extra = pd.DataFrame(rows, index=table.index)
    t = table.drop(columns=[c for c in extra.columns if c in table.columns]).join(extra)
    for col in ["TT"] + SIGNALS:
        t[col] = t[col].fillna(False).astype(bool)
    quality = QUALITY if use_fundamentals else QUALITY_TECH
    t["n_signals"] = t[SIGNALS].sum(axis=1).astype(int)
    t["swing_score"] = (t[quality].sum(axis=1) + (t["n_signals"] > 0)).astype(int)
    t["swing_max"] = len(quality) + 1
    t["swing_candidate"] = t[quality].all(axis=1) & (t["n_signals"] > 0) & market_pass
    return t.sort_values(["swing_score", "n_signals", "rs_rank"], ascending=False,
                         kind="stable").reset_index(drop=True)


# ---- historical record of the signals ----------------------------------------

def simulate_trade(o, h, lo, c, i: int, risk: float, cfg: SwingConfig):
    """Trade entered at the open of bar i+1 after a signal on bar i.

    Returns (entry, stop, target, exit_index, exit_price, reason) where reason
    is "stop", "target" or "time"; exit_index/exit_price/reason are None
    while the trade is still open at the end of the data. None when there is
    no entry bar yet or the risk is unusable.
    """
    n, e = len(c), i + 1
    if e >= n or not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[e]
    r = risk * entry                         # risk as a fraction of price
    stop, target = entry - r, entry + cfg.target_r * r
    last = e + cfg.max_hold - 1
    for j in range(e, min(n - 1, last) + 1):
        if j > e and o[j] <= stop:           # gapped through the stop
            return entry, stop, target, j, o[j], "stop"
        if lo[j] <= stop:                    # stop first when both touch
            return entry, stop, target, j, stop, "stop"
        if j > e and o[j] >= target:
            return entry, stop, target, j, o[j], "target"
        if h[j] >= target:
            return entry, stop, target, j, target, "target"
    if last <= n - 1:
        return entry, stop, target, last, c[last], "time"
    return entry, stop, target, None, None, None


def _simulate(o, h, lo, c, entries, risk, cfg: SwingConfig, cost: float) -> np.ndarray:
    """R multiple of each closed trade entered at the open after a signal bar."""
    out = []
    for i in entries:
        t = simulate_trade(o, h, lo, c, i, risk[i], cfg)
        if t is None or t[3] is None:
            continue                         # no entry yet, or still open
        entry, stop = t[0], t[1]
        out.append((t[4] - entry - cost * entry) / (entry - stop))
    return np.array(out, float)


def signal_stats(price_map: dict[str, pd.DataFrame], cfg: SwingConfig | None = None,
                 crypto: bool = False, baseline_step: int = 5) -> dict:
    """Replay the trade plan after every signal across all symbols.

    Returns per rule: trades, win rate, average R, and the average R of
    entering every ``baseline_step``-th bar ("any day") for comparison.
    Only bars after the 252-bar warm-up are used.
    """
    cfg = cfg or SwingConfig()
    cost = cfg.cost_crypto if crypto else cfg.cost_stock
    rules = {k: [] for k in ["BASE", "TT", *SIGNALS, "TT+SIG"]}
    first = last = None
    for data in price_map.values():
        if len(data) < 252 + cfg.max_hold + 2:
            continue
        sig = signal_frame(data, cfg)
        risk = np.full(len(data), cfg.stop_pct)
        o, h, lo, c = (data[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        start = 252
        first = min(first or data.index[start], data.index[start])
        last = max(last or data.index[-1], data.index[-1])
        idx = np.arange(start, len(data))
        masks = {"BASE": (idx % baseline_step) == 0, "TT": sig["TT"].to_numpy()[idx]}
        for k in SIGNALS:
            masks[k] = sig[k].to_numpy()[idx]
        masks["TT+SIG"] = masks["TT"] & sig[SIGNALS].any(axis=1).to_numpy()[idx]
        for k, m in masks.items():
            rules[k].append(_simulate(o, h, lo, c, idx[m], risk, cfg, cost))
    out = {}
    for k, parts in rules.items():
        r = np.concatenate(parts) if parts else np.array([])
        out[k] = {"n": int(len(r)),
                  "win": float((r > 0).mean()) if len(r) else None,
                  "avg_r": float(r.mean()) if len(r) else None}
    return {"rules": out, "symbols": len(price_map),
            "from": first.strftime("%Y-%m-%d") if first is not None else None,
            "to": last.strftime("%Y-%m-%d") if last is not None else None,
            "cost": cost, "stop_pct": cfg.stop_pct, "target_pct": cfg.target_pct,
            "max_hold": cfg.max_hold}
