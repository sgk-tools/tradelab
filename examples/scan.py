"""Walk-forward scan: every strategy on every downloaded market.

Usage: python examples/scan.py
Reads data/*.csv (create them with `python -m tradelab fetch ...`).

Caution: this runs many strategy/market pairs. Picking the best row afterwards
is itself a form of over-fitting; treat the table as a screen, not as proof.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelab import (backtest, compute_metrics, get_cost_model, load_csv,  # noqa: E402
                      make_signal, walk_forward)

JPY = 300_000.0
USDJPY = load_csv(ROOT / "data/usdjpy.csv")["close"].iloc[-1]

# file, cost preset, capital in the instrument's quote currency
MARKETS = {
    "Toyota 7203 (JP stock)": ("toyota.csv", "jp_stock", JPY),
    "SPY (US stock ETF)": ("spy.csv", "us_stock", JPY / USDJPY),
    "USD/JPY (FX)": ("usdjpy.csv", "fx", JPY),
    "BTC/USD (crypto)": ("btcusd.csv", "crypto", JPY / USDJPY),
}
GRIDS = {
    "sma_cross": {"fast": [10, 20, 50], "slow": [100, 150, 200]},
    "donchian_breakout": {"entry": [20, 55, 100], "exit": [10, 20, 50]},
    "rsi_reversion": {"period": [7, 14], "lower": [20, 30], "exit_level": [50, 60]},
    "momentum": {"lookback": [60, 120, 250]},
}
TRAIN, TEST = 750, 250

rows = []
for market, (file, preset, capital) in MARKETS.items():
    data = load_csv(ROOT / "data" / file)
    cost = get_cost_model(preset)
    oos = data.iloc[TRAIN:]
    b = backtest(oos, make_signal("buy_and_hold", oos), cost, capital)
    bm = compute_metrics(b.equity, b.trades, capital, b.units)
    rows.append({"market": market, "strategy": "buy_and_hold", **bm,
                 "skipped": b.stats["skipped_orders"], "pos_folds": ""})
    for name, grid in GRIDS.items():
        if cost.allow_short:
            grid = {**grid, "allow_short": [False, True]}
        wf = walk_forward(data, name, grid, cost, capital, TRAIN, TEST)
        m = compute_metrics(wf.equity, wf.trades, capital, wf.units)
        f = wf.folds
        rows.append({"market": market, "strategy": name, **m, "skipped": "",
                     "pos_folds": f"{int((f['oos_return'] > 0).sum())}/{len(f)}"})

df = pd.DataFrame(rows)
out = df[["market", "strategy"]].copy()
out["CAGR"] = (df["cagr"] * 100).map("{:+.1f}%".format)
out["Sharpe"] = df["sharpe"].map("{:.2f}".format)
out["MaxDD"] = (df["max_drawdown"] * 100).map("{:.1f}%".format)
out["trades"] = df["trades"]
out["exposure"] = (df["exposure"] * 100).map("{:.0f}%".format)
out["profitable folds"] = df["pos_folds"]
out["unaffordable orders"] = df["skipped"]
print(f"out-of-sample period starts after {TRAIN} training bars; "
      f"train={TRAIN} test={TEST}; capital = JPY {JPY:,.0f} "
      f"(= USD {JPY / USDJPY:,.0f} at {USDJPY:.2f})")
print(out.to_string(index=False))
df.to_csv(ROOT / "reports/scan_summary.csv", index=False)
