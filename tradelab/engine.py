"""Bar-by-bar backtest engine for a single instrument.

Timing convention (prevents look-ahead bias):
    * ``signal[t]`` is the target exposure decided with information available
      at the CLOSE of bar t.
    * The resulting order is filled at the OPEN of bar t+1, at a price made
      worse by ``slippage_rate``, plus commission.
    * Equity is marked to the close of every bar.

A signal value is the target position as a fraction of equity: +1 fully long,
0 flat, -1 fully short. Orders are only sent when the target changes, so a
constant signal does not generate rebalancing trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel

_TRADE_COLS = ["entry_time", "exit_time", "direction", "units", "entry_price",
               "exit_price", "pnl", "return", "bars"]


@dataclass
class BacktestResult:
    equity: pd.Series            # account value at each bar close
    units: pd.Series             # position (signed units) at each bar close
    trades: pd.DataFrame         # closed round-trip trades
    initial_capital: float
    cost: CostModel
    stats: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])


def _floor_lot(units: float, lot: float) -> float:
    if lot <= 0:
        return max(units, 0.0)
    return math.floor(units / lot + 1e-9) * lot


def backtest(data: pd.DataFrame, signal: pd.Series, cost: CostModel | None = None,
             initial_capital: float = 300_000.0,
             max_drawdown_stop: float | None = None) -> BacktestResult:
    """Simulate trading ``signal`` on ``data``.

    max_drawdown_stop: e.g. 0.3 liquidates and stops trading for good once
    equity has fallen 30% from its peak (checked at bar close, executed at the
    next open). ``None`` disables the kill switch.
    """
    cost = cost or CostModel()
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if max_drawdown_stop is not None and not 0 < max_drawdown_stop < 1:
        raise ValueError("max_drawdown_stop must be between 0 and 1")
    for col in ("open", "close"):
        if col not in data.columns:
            raise ValueError(f"data is missing column '{col}'")
    if len(data) < 2:
        raise ValueError("need at least 2 bars")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("data index must be sorted and unique")

    idx = data.index
    n = len(idx)
    opens = data["open"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    sig = signal.reindex(idx).fillna(0.0).to_numpy(float)
    low_bound = -cost.max_leverage if cost.allow_short else 0.0
    sig = np.clip(sig, low_bound, cost.max_leverage)
    dt_years = np.zeros(n)
    dt_years[1:] = np.diff(idx.values).astype("timedelta64[s]").astype(float) \
        / (365.25 * 86400)

    cash = float(initial_capital)
    units = 0.0
    prev_target = 0.0
    peak = cash
    halted = False
    halt_time = None
    commission_paid = slippage_paid = financing_paid = 0.0
    skipped_orders = 0
    open_trade: dict | None = None
    trades: list[dict] = []
    equity = np.empty(n)
    pos = np.empty(n)

    def fill(delta: float, price: float) -> float:
        """Execute ``delta`` units at the bar open. Returns the fill price."""
        nonlocal cash, units, commission_paid, slippage_paid
        side = 1.0 if delta > 0 else -1.0
        px = price * (1 + cost.slippage_rate * side)
        notional = abs(delta) * px
        comm = max(cost.min_commission, notional * cost.commission_rate)
        cash -= delta * px + comm
        units += delta
        commission_paid += comm
        slippage_paid += abs(delta) * price * cost.slippage_rate
        return px

    def size(equity_now: float, target: float, price: float) -> float:
        """Units for ``target`` exposure such that cost fits in the budget."""
        if equity_now <= 0 or target == 0:
            return 0.0
        side = 1.0 if target > 0 else -1.0
        px = price * (1 + cost.slippage_rate * side)
        budget = equity_now * abs(target)
        comm = max(cost.min_commission,
                   budget * cost.commission_rate / (1 + cost.commission_rate))
        return side * _floor_lot((budget - comm) / px, cost.lot_size)

    def close_position(i: int) -> None:
        nonlocal units, open_trade
        px = fill(-units, opens[i])
        units = 0.0
        t = open_trade
        pnl = cash - t["cash_before"]
        trades.append({
            "entry_time": idx[t["i"]], "exit_time": idx[i],
            "direction": t["direction"], "units": t["units"],
            "entry_price": t["price"], "exit_price": px, "pnl": pnl,
            "return": pnl / t["cash_before"] if t["cash_before"] > 0 else np.nan,
            "bars": i - t["i"],
        })
        open_trade = None

    for i in range(n):
        if i > 0:
            # Financing on the position carried since the previous bar.
            if cost.annual_borrow_rate and units != 0:
                borrowed = max(0.0, -cash) if units > 0 else -units * closes[i - 1]
                fee = borrowed * cost.annual_borrow_rate * dt_years[i]
                cash -= fee
                financing_paid += fee

            target = 0.0 if halted else sig[i - 1]
            if target != prev_target or (halted and units != 0):
                price = opens[i]
                if units != 0 and (target == 0 or (units > 0) != (target > 0)):
                    close_position(i)
                if target != 0:
                    desired = size(cash + units * price, target, price)
                    if desired == 0:
                        if units != 0:
                            close_position(i)
                        else:
                            skipped_orders += 1
                    elif desired != units:
                        if units == 0:
                            open_trade = {"i": i, "cash_before": cash,
                                          "direction": "long" if desired > 0 else "short",
                                          "units": abs(desired)}
                            open_trade["price"] = fill(desired, price)
                        else:
                            fill(desired - units, price)
                prev_target = target

        eq = cash + units * closes[i]
        equity[i] = eq
        pos[i] = units
        peak = max(peak, eq)
        if not halted:
            hit_stop = (max_drawdown_stop is not None
                        and eq <= peak * (1 - max_drawdown_stop))
            if hit_stop or eq <= 0:
                halted = True
                halt_time = idx[i]

    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS)
    stats = {
        "commission_paid": commission_paid,
        "slippage_paid": slippage_paid,
        "financing_paid": financing_paid,
        "total_cost": commission_paid + slippage_paid + financing_paid,
        "skipped_orders": skipped_orders,
        "halted": halted,
        "halt_time": halt_time,
        "open_position_at_end": units != 0,
    }
    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        units=pd.Series(pos, index=idx, name="units"),
        trades=trades_df, initial_capital=float(initial_capital),
        cost=cost, stats=stats)
