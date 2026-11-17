"""Trading strategies.

Every strategy maps OHLCV data to a target-exposure Series (+1 long, 0 flat,
-1 short) and must be CAUSAL: the value at bar t may only depend on bars
<= t. ``tests/test_strategies.py`` verifies this for all registered
strategies, so register new ones in ``STRATEGIES`` to get the same check.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _latch(index: pd.Index, enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """1 from an ``enter`` bar until the next ``exit_`` bar, else 0."""
    state = pd.Series(np.nan, index=index)
    state[enter] = 1.0
    state[exit_ & ~enter] = 0.0
    return state.ffill().fillna(0.0)


def buy_and_hold(data: pd.DataFrame) -> pd.Series:
    """Always fully long. The benchmark every strategy must beat."""
    return pd.Series(1.0, index=data.index)


def sma_cross(data: pd.DataFrame, fast: int = 20, slow: int = 100,
              allow_short: bool = False) -> pd.Series:
    """Trend following: long while SMA(fast) > SMA(slow)."""
    if not 1 <= fast < slow:
        raise ValueError("require 1 <= fast < slow")
    close = data["close"]
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    sig = pd.Series(0.0, index=data.index)
    sig[f > s] = 1.0
    if allow_short:
        sig[f < s] = -1.0
    sig[s.isna()] = 0.0
    return sig


def donchian_breakout(data: pd.DataFrame, entry: int = 55, exit: int = 20,
                      allow_short: bool = False) -> pd.Series:
    """Breakout: long on a close above the prior ``entry``-bar high,
    exit on a close below the prior ``exit``-bar low (mirrored for shorts)."""
    if entry < 2 or exit < 1:
        raise ValueError("require entry >= 2 and exit >= 1")
    close = data["close"]
    hi_entry = data["high"].rolling(entry).max().shift(1)
    lo_exit = data["low"].rolling(exit).min().shift(1)
    sig = _latch(data.index, close > hi_entry, close < lo_exit)
    if allow_short:
        lo_entry = data["low"].rolling(entry).min().shift(1)
        hi_exit = data["high"].rolling(exit).max().shift(1)
        sig = sig - _latch(data.index, close < lo_entry, close > hi_exit)
    return sig


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + up / down)
    return out.where(down != 0, 100.0).where(up.notna())


def rsi_reversion(data: pd.DataFrame, period: int = 14, lower: float = 30,
                  upper: float = 70, exit_level: float = 50,
                  allow_short: bool = False) -> pd.Series:
    """Mean reversion: long when RSI < lower, exit when RSI > exit_level
    (mirrored for shorts above ``upper``)."""
    if period < 2 or not 0 < lower < exit_level < upper < 100:
        raise ValueError("require period >= 2 and 0 < lower < exit_level < upper < 100")
    r = rsi(data["close"], period)
    sig = _latch(data.index, r < lower, r > exit_level)
    if allow_short:
        sig = sig - _latch(data.index, r > upper, r < exit_level)
    return sig


def momentum(data: pd.DataFrame, lookback: int = 120,
             allow_short: bool = False) -> pd.Series:
    """Time-series momentum: long while the ``lookback``-bar return is > 0."""
    if lookback < 1:
        raise ValueError("require lookback >= 1")
    ret = data["close"].pct_change(lookback)
    sig = pd.Series(0.0, index=data.index)
    sig[ret > 0] = 1.0
    if allow_short:
        sig[ret < 0] = -1.0
    return sig


STRATEGIES: dict[str, Callable[..., pd.Series]] = {
    "buy_and_hold": buy_and_hold,
    "sma_cross": sma_cross,
    "donchian_breakout": donchian_breakout,
    "rsi_reversion": rsi_reversion,
    "momentum": momentum,
}


def make_signal(strategy: str | Callable[..., pd.Series], data: pd.DataFrame,
                **params) -> pd.Series:
    """Build a signal from a registered strategy name or a callable."""
    if isinstance(strategy, str):
        try:
            fn = STRATEGIES[strategy]
        except KeyError:
            raise ValueError(f"unknown strategy '{strategy}'. "
                             f"choices: {', '.join(STRATEGIES)}") from None
    else:
        fn = strategy
    return fn(data, **params)


# --------------------------------------------------------------------------
# Dow theory
# --------------------------------------------------------------------------

def swing_points(data: pd.DataFrame, swing: int = 5) -> pd.DataFrame:
    """Confirmed swing highs/lows.

    A bar is a swing high if its high is the highest of the ``swing`` bars on
    each side. It can only be known ``swing`` bars later, so the value is
    reported at the confirmation bar (NaN elsewhere). This keeps it causal.
    """
    if swing < 1:
        raise ValueError("require swing >= 1")
    w = 2 * swing + 1
    hi, lo = data["high"].shift(swing), data["low"].shift(swing)
    return pd.DataFrame({
        "swing_high": hi.where(hi == data["high"].rolling(w).max()),
        "swing_low": lo.where(lo == data["low"].rolling(w).min()),
    }, index=data.index)


def dow_theory(data: pd.DataFrame, swing: int = 5,
               allow_short: bool = False) -> pd.Series:
    """Dow theory trend: +1 uptrend, -1 downtrend, 0 no trend.

    Uptrend starts when the latest swing low is higher than the one before
    (higher low) and the close breaks above the latest swing high (higher
    high). It lasts until a close below the latest swing low. Downtrend is
    the mirror image. Without ``allow_short`` a downtrend is reported as 0.
    """
    sp = swing_points(data, swing)
    sh, sl = sp["swing_high"].to_numpy(), sp["swing_low"].to_numpy()
    close = data["close"].to_numpy(float)
    nan = float("nan")
    h_prev = h_last = l_prev = l_last = nan
    trend = 0
    out = np.zeros(len(close))
    for t in range(len(close)):
        if sh[t] == sh[t]:
            h_prev, h_last = h_last, sh[t]
        if sl[t] == sl[t]:
            l_prev, l_last = l_last, sl[t]
        c = close[t]
        if trend == 1 and c < l_last:
            trend = 0
        elif trend == -1 and c > h_last:
            trend = 0
        if trend != 1 and l_last > l_prev and c > h_last:
            trend = 1
        elif trend != -1 and h_last < h_prev and c < l_last:
            trend = -1
        out[t] = trend
    sig = pd.Series(out, index=data.index)
    return sig if allow_short else sig.clip(lower=0)


STRATEGIES["dow_theory"] = dow_theory
