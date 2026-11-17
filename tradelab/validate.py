"""Parameter search and walk-forward (out-of-sample) validation.

A parameter set chosen because it scored best on past data is biased upward:
the more combinations tried, the better the best one looks by chance alone.
Walk-forward validation measures that: parameters are selected on a training
window, then scored on the following window that played no part in the
selection. Only the stitched out-of-sample result is an honest estimate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .costs import CostModel
from .engine import backtest
from .metrics import compute_metrics
from .strategies import make_signal


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(param_grid[k] for k in keys))]


def _signals(data, strategy, param_grid) -> list[tuple[dict, pd.Series]]:
    """Signals for every valid parameter combination (invalid ones skipped)."""
    out = []
    for params in expand_grid(param_grid):
        try:
            out.append((params, make_signal(strategy, data, **params)))
        except ValueError:
            continue
    if not out:
        raise ValueError("no valid parameter combination in the grid")
    return out


def _score(metrics: dict, metric: str, min_trades: int) -> float:
    value = metrics.get(metric, np.nan)
    if metrics.get("trades", 0) < min_trades or not np.isfinite(value):
        return -np.inf
    return float(value)


def grid_search(data: pd.DataFrame, strategy: str | Callable,
                param_grid: dict[str, list], cost: CostModel | None = None,
                initial_capital: float = 300_000.0, metric: str = "sharpe",
                min_trades: int = 1) -> pd.DataFrame:
    """Backtest every combination on ``data``. IN-SAMPLE: do not treat the
    top row as an expected future result."""
    rows = []
    for params, sig in _signals(data, strategy, param_grid):
        res = backtest(data, sig, cost, initial_capital)
        m = compute_metrics(res.equity, res.trades, initial_capital, res.units)
        rows.append({**params, **m, "score": _score(m, metric, min_trades)})
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)


@dataclass
class WalkForwardResult:
    equity: pd.Series        # stitched out-of-sample equity curve
    units: pd.Series
    trades: pd.DataFrame
    folds: pd.DataFrame      # per-fold chosen params, in-sample vs OOS score
    initial_capital: float
    n_combinations: int
    metric: str


def walk_forward(data: pd.DataFrame, strategy: str | Callable,
                 param_grid: dict[str, list], cost: CostModel | None = None,
                 initial_capital: float = 300_000.0, train_bars: int = 500,
                 test_bars: int = 125, metric: str = "sharpe",
                 min_trades: int = 1) -> WalkForwardResult:
    """Rolling walk-forward optimisation.

    For each fold: pick the best parameters on ``train_bars`` bars, trade them
    on the next ``test_bars`` bars, then roll forward by ``test_bars``. Capital
    carries over between folds; positions are closed at fold boundaries by
    marking to the last close (no extra exit cost is charged there).
    """
    n = len(data)
    if train_bars < 2 or test_bars < 2:
        raise ValueError("train_bars and test_bars must be >= 2")
    if n < train_bars + test_bars:
        raise ValueError(
            f"need at least train_bars + test_bars = {train_bars + test_bars} "
            f"bars, got {n}")

    candidates = _signals(data, strategy, param_grid)
    capital = float(initial_capital)
    eq_parts, unit_parts, trade_parts, folds = [], [], [], []

    start = 0
    while start + train_bars + 2 <= n:
        tr = slice(start, start + train_bars)
        te = slice(start + train_bars, min(start + train_bars + test_bars, n))
        train, test = data.iloc[tr], data.iloc[te]

        best, best_score = None, -np.inf
        for params, sig in candidates:
            r = backtest(train, sig.iloc[tr], cost, initial_capital)
            s = _score(compute_metrics(r.equity, r.trades, initial_capital),
                       metric, min_trades)
            if s > best_score:
                best, best_score = (params, sig), s

        if best is None:        # nothing traded in training: stay flat
            params, sig = {}, pd.Series(0.0, index=data.index)
        else:
            params, sig = best
        r = backtest(test, sig.iloc[te], cost, capital)
        m = compute_metrics(r.equity, r.trades, capital)
        folds.append({
            "train_start": train.index[0], "train_end": train.index[-1],
            "test_start": test.index[0], "test_end": test.index[-1],
            "params": params, "in_sample_score": best_score,
            "oos_score": m.get(metric, np.nan),
            "oos_return": m["total_return"], "oos_trades": m["trades"],
        })
        eq_parts.append(r.equity)
        unit_parts.append(r.units)
        trade_parts.append(r.trades)
        capital = r.final_equity
        if capital <= 0:
            break
        start += test_bars

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return WalkForwardResult(
        equity=pd.concat(eq_parts), units=pd.concat(unit_parts), trades=trades,
        folds=pd.DataFrame(folds), initial_capital=float(initial_capital),
        n_combinations=len(candidates), metric=metric)
