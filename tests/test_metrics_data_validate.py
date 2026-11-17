import numpy as np
import pandas as pd
import pytest

from tradelab import (compute_metrics, get_cost_model, grid_search, load_csv,
                      synthetic_ohlcv, walk_forward)
from tradelab.data import DataError, validate_ohlcv
from tradelab.metrics import max_drawdown


# ---- metrics --------------------------------------------------------------

def test_max_drawdown_known_value():
    eq = pd.Series([100, 120, 90, 95, 60, 130], dtype=float)
    mdd, bars = max_drawdown(eq)
    assert mdd == pytest.approx(60 / 120 - 1) and bars == 3


def test_cagr_and_total_return():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    m = compute_metrics(pd.Series([100, 110, 121.0], index=idx), initial_capital=100)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["cagr"] == pytest.approx(0.10, abs=2e-4)


def test_trade_statistics():
    idx = pd.bdate_range("2024-01-01", periods=5)
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 20.0, -5.0], "bars": [1, 2, 3, 2]})
    m = compute_metrics(pd.Series(np.linspace(100, 120, 5), index=idx), trades, 100)
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(3.0)


def test_flat_equity_has_no_sharpe():
    idx = pd.bdate_range("2024-01-01", periods=10)
    m = compute_metrics(pd.Series(100.0, index=idx), pd.DataFrame(columns=["pnl", "bars"]))
    assert np.isnan(m["sharpe"]) and m["max_drawdown"] == 0 and m["trades"] == 0


# ---- data -----------------------------------------------------------------

def test_load_csv_japanese_headers_and_sorting(tmp_path):
    p = tmp_path / "jp.csv"
    p.write_text("日付,始値,高値,安値,終値,出来高\n"
                 "2024-01-05,103,106,102,105,1000\n"
                 "2024-01-04,100,104,99,103,900\n"
                 "2024-01-04,100,104,99,103,900\n", encoding="utf-8")
    d = load_csv(p)
    assert list(d.columns) == ["open", "high", "low", "close", "volume"]
    assert len(d) == 2 and d.index.is_monotonic_increasing
    assert d["close"].iloc[-1] == 105


def test_load_csv_applies_adjusted_close(tmp_path):
    p = tmp_path / "adj.csv"
    p.write_text("Date,Open,High,Low,Close,Adj Close,Volume\n"
                 "2024-01-04,200,220,190,210,105,1\n"
                 "2024-01-05,100,110,95,105,105,1\n")
    d = load_csv(p)
    assert d["close"].tolist() == [105, 105] and d["open"].iloc[0] == 100
    assert load_csv(p, adjust=False)["close"].tolist() == [210, 105]


@pytest.mark.parametrize("row", [
    "2024-01-05,100,90,95,98,1",      # high below open
    "2024-01-05,100,110,105,108,1",   # low above open
    "2024-01-05,-1,110,-2,108,1",     # negative price
])
def test_invalid_bars_rejected(tmp_path, row):
    p = tmp_path / "bad.csv"
    p.write_text("date,open,high,low,close,volume\n2024-01-04,100,104,99,103,1\n" + row + "\n")
    with pytest.raises(DataError):
        load_csv(p)


def _frame(high):
    idx = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame({"open": [100.0, 100, 100], "high": [101.0, high, 101],
                         "low": [99.0, 99, 99], "close": [100.0, 100, 100]}, index=idx)


def test_small_inconsistency_repaired_only_within_tolerance():
    d = validate_ohlcv(_frame(99.9), repair_tolerance=0.005)   # off by 0.1%
    assert d.attrs["repaired_bars"] == 1 and d["high"].iloc[1] == 100
    assert validate_ohlcv(_frame(101)).attrs["repaired_bars"] == 0
    with pytest.raises(DataError):
        validate_ohlcv(_frame(99.9))                            # strict default
    with pytest.raises(DataError):
        validate_ohlcv(_frame(95.0), repair_tolerance=0.005)    # off by 5%


def test_missing_columns_rejected():
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(DataError):
        validate_ohlcv(pd.DataFrame({"close": [1, 2, 3]}, index=idx))


def test_synthetic_is_valid_and_reproducible():
    a, b = synthetic_ohlcv(300, seed=7), synthetic_ohlcv(300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    validate_ohlcv(a)


# ---- validation -----------------------------------------------------------

GRID = {"fast": [5, 10, 20], "slow": [20, 50]}


def test_grid_search_skips_invalid_and_sorts():
    g = grid_search(synthetic_ohlcv(600, seed=3), "sma_cross", GRID, get_cost_model("zero"))
    assert len(g) == 5                       # fast=20, slow=20 is invalid
    assert g["score"].is_monotonic_decreasing


def test_walk_forward_folds_are_out_of_sample_and_contiguous():
    d = synthetic_ohlcv(1000, seed=4)
    wf = walk_forward(d, "sma_cross", GRID, get_cost_model("crypto"),
                      train_bars=300, test_bars=100)
    f = wf.folds
    assert len(f) == 7 and wf.n_combinations == 5
    assert (f["train_end"] < f["test_start"]).all()
    assert (f["test_end"].iloc[:-1].values < f["test_start"].iloc[1:].values).all()
    assert wf.equity.index.equals(d.index[300:])        # only unseen bars
    assert wf.equity.index.is_unique


def test_walk_forward_capital_carries_between_folds():
    d = synthetic_ohlcv(800, seed=5)
    wf = walk_forward(d, "sma_cross", GRID, get_cost_model("zero"),
                      initial_capital=1000, train_bars=300, test_bars=100)
    first_fold_end = wf.equity.loc[wf.folds["test_end"].iloc[0]]
    second_fold_start = wf.equity.loc[wf.folds["test_start"].iloc[1]]
    assert second_fold_start == pytest.approx(first_fold_end)


def test_walk_forward_does_not_use_test_data_for_selection():
    """Changing prices inside a test window must not change the parameters
    chosen for that window."""
    d = synthetic_ohlcv(700, seed=6)
    kw = dict(cost=get_cost_model("zero"), train_bars=400, test_bars=150)
    base = walk_forward(d, "sma_cross", GRID, **kw)
    altered = d.copy()
    altered.iloc[400:550] = altered.iloc[400:550] * 3
    alt = walk_forward(altered, "sma_cross", GRID, **kw)
    assert base.folds["params"].iloc[0] == alt.folds["params"].iloc[0]


def test_walk_forward_needs_enough_data():
    with pytest.raises(ValueError):
        walk_forward(synthetic_ohlcv(100), "sma_cross", GRID, train_bars=300, test_bars=100)


def test_no_edge_on_driftless_random_walk():
    """With costs and no drift, trend following must not make money on
    average. Guards against an engine bug that manufactures profit."""
    cost = get_cost_model("crypto")
    finals = []
    for seed in range(40):
        d = synthetic_ohlcv(750, seed=seed, drift=0.0, vol=0.3)
        wf = walk_forward(d, "sma_cross", GRID, cost, initial_capital=1000,
                          train_bars=250, test_bars=125)
        finals.append(wf.equity.iloc[-1] / 1000 - 1)
    assert np.mean(finals) < 0.02


def test_forming_daily_bar_is_dropped_only_while_session_is_open():
    from datetime import datetime, timezone

    from tradelab.data import drop_forming_bar
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-11-16", "2026-11-17"]))
    close = datetime(2026, 11, 17, 21, 0, tzinfo=timezone.utc).timestamp()
    stock = {"instrumentType": "EQUITY", "currentTradingPeriod": {"regular": {"end": close}}}
    during = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
    after = datetime(2026, 11, 17, 22, 13, tzinfo=timezone.utc)
    next_day = datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc)
    assert len(drop_forming_bar(df, stock, during)) == 1
    assert len(drop_forming_bar(df, stock, after)) == 2
    crypto = {"instrumentType": "CRYPTOCURRENCY", "currentTradingPeriod": {"regular": {"end": close}}}
    assert len(drop_forming_bar(df, crypto, after)) == 1
    assert len(drop_forming_bar(df, crypto, next_day)) == 2
    assert len(drop_forming_bar(df, {}, after)) == 1          # unknown: be conservative
