"""Performance metrics computed from an equity curve and a trade list."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Largest peak-to-trough loss (negative fraction) and the longest
    number of bars spent below a previous peak."""
    peak = equity.cummax()
    dd = equity / peak - 1
    under = (dd < 0).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None,
                    initial_capital: float | None = None,
                    units: pd.Series | None = None) -> dict:
    """Summary statistics. Risk-free rate is assumed to be 0."""
    initial = float(initial_capital if initial_capital is not None else equity.iloc[0])
    final = float(equity.iloc[-1])
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    bars_per_year = len(rets) / years if years > 0 else np.nan

    total_return = final / initial - 1
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 and final > 0 else np.nan
    std = rets.std()
    vol = std * np.sqrt(bars_per_year) if len(rets) > 1 else np.nan
    sharpe = rets.mean() / std * np.sqrt(bars_per_year) if std and std > 0 else np.nan
    downside = np.sqrt((rets.clip(upper=0) ** 2).mean()) if len(rets) else np.nan
    sortino = rets.mean() / downside * np.sqrt(bars_per_year) \
        if downside and downside > 0 else np.nan
    mdd, mdd_bars = max_drawdown(pd.concat(
        [pd.Series([initial]), equity.reset_index(drop=True)], ignore_index=True))
    calmar = cagr / abs(mdd) if mdd < 0 and np.isfinite(cagr) else np.nan

    out = {
        "initial_capital": initial, "final_equity": final,
        "total_return": total_return, "cagr": cagr, "volatility": vol,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd,
        "max_drawdown_bars": mdd_bars, "calmar": calmar, "years": years,
        "bars": len(equity),
    }
    if units is not None:
        out["exposure"] = float((units != 0).mean())
    if trades is not None:
        n = len(trades)
        out["trades"] = n
        if n:
            pnl = trades["pnl"]
            wins, losses = pnl[pnl > 0], pnl[pnl < 0]
            out["win_rate"] = len(wins) / n
            out["profit_factor"] = float(wins.sum() / -losses.sum()) \
                if len(losses) else np.inf
            out["avg_trade_pnl"] = float(pnl.mean())
            out["avg_bars_held"] = float(trades["bars"].mean())
        else:
            out.update(win_rate=np.nan, profit_factor=np.nan,
                       avg_trade_pnl=np.nan, avg_bars_held=np.nan)
    return out
