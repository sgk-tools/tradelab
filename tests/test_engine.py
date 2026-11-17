import numpy as np
import pandas as pd
import pytest

from tradelab import CostModel, backtest, synthetic_ohlcv


def bars(opens, closes=None):
    opens = np.asarray(opens, float)
    closes = np.asarray(closes if closes is not None else opens, float)
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes),
                         "low": np.minimum(opens, closes), "close": closes}, index=idx)


def sig(data, values):
    return pd.Series(values, index=data.index, dtype=float)


def test_flat_signal_keeps_capital():
    d = synthetic_ohlcv(200, seed=1)
    r = backtest(d, sig(d, 0.0), CostModel(commission_rate=0.01), 1000)
    assert (r.equity == 1000).all() and r.trades.empty
    assert r.stats["total_cost"] == 0


def test_order_fills_at_next_open_not_signal_bar():
    # Signal appears at close of bar 1 -> must fill at open of bar 2 (=120).
    d = bars([100, 110, 120, 130], [100, 110, 120, 130])
    r = backtest(d, sig(d, [0, 1, 1, 1]), CostModel(), 1200)
    assert list(r.units) == [0, 0, 10, 10]
    assert r.equity.iloc[1] == 1200          # no profit from bar 1 -> 2 gap
    assert r.equity.iloc[-1] == pytest.approx(1300)


def test_last_bar_signal_is_never_executed():
    d = bars([100, 100, 100])
    r = backtest(d, sig(d, [0, 0, 1]), CostModel(), 1000)
    assert (r.units == 0).all()


def test_zero_cost_buy_and_hold_tracks_price():
    d = synthetic_ohlcv(300, seed=2)
    r = backtest(d, sig(d, 1.0), CostModel(), 1000)
    expected = 1000 * d["close"].iloc[-1] / d["open"].iloc[1]
    assert r.final_equity == pytest.approx(expected)


def test_round_trip_costs_by_hand():
    d = bars([100, 100, 100, 110, 110], [100, 100, 100, 110, 110])
    cost = CostModel(commission_rate=0.01, slippage_rate=0.01, lot_size=1)
    r = backtest(d, sig(d, [1, 1, 0, 0, 0]), cost, 10_000)
    # Buy at 101: budget 10000, commission reserve 10000*0.01/1.01 = 99.0099
    # -> floor(9900.99/101) = 98 units; commission 98*101*0.01 = 98.98
    # Sell at open of bar 3 = 110*0.99 = 108.9; commission 98*108.9*0.01
    cash = 10_000 - 98 * 101 - 98.98
    cash += 98 * 108.9 - 98 * 108.9 * 0.01
    assert r.final_equity == pytest.approx(cash)
    t = r.trades.iloc[0]
    assert len(r.trades) == 1 and t["units"] == 98 and t["bars"] == 2
    assert t["entry_price"] == pytest.approx(101) and t["exit_price"] == pytest.approx(108.9)
    assert t["pnl"] == pytest.approx(cash - 10_000)
    assert r.stats["commission_paid"] == pytest.approx(98.98 + 98 * 108.9 * 0.01)


def test_costs_reduce_result():
    d = synthetic_ohlcv(500, seed=3)
    s = sig(d, np.where(np.arange(500) // 5 % 2 == 0, 1.0, 0.0))
    free = backtest(d, s, CostModel(), 1000).final_equity
    paid = backtest(d, s, CostModel(commission_rate=0.001, slippage_rate=0.001), 1000)
    assert paid.final_equity < free and paid.stats["total_cost"] > 0


def test_lot_size_rounding_and_unaffordable_order():
    d = bars([100, 100, 100, 100])
    r = backtest(d, sig(d, 1.0), CostModel(lot_size=100), 25_050)
    assert r.units.iloc[-1] == 200
    r = backtest(d, sig(d, 1.0), CostModel(lot_size=100), 9_999)
    assert (r.units == 0).all() and r.stats["skipped_orders"] == 1


def test_minimum_commission():
    d = bars([100, 100, 100, 100])
    r = backtest(d, sig(d, [1, 0, 0, 0]), CostModel(min_commission=50, lot_size=1), 1000)
    assert r.final_equity == pytest.approx(900)      # 50 in + 50 out


def test_cash_never_negative_without_leverage():
    d = synthetic_ohlcv(400, seed=4)
    s = sig(d, np.where(np.arange(400) // 7 % 2 == 0, 1.0, 0.0))
    cost = CostModel(commission_rate=0.005, slippage_rate=0.002, min_commission=5, lot_size=1)
    r = backtest(d, s, cost, 50_000)
    cash = r.equity - r.units * d["close"]
    assert cash.min() >= -1e-6


def test_short_disallowed_is_clipped():
    d = bars([100, 100, 90, 80])
    r = backtest(d, sig(d, -1.0), CostModel(allow_short=False), 1000)
    assert (r.units == 0).all() and r.final_equity == 1000


def test_short_profits_when_price_falls():
    d = bars([100, 100, 90, 80], [100, 100, 90, 80])
    r = backtest(d, sig(d, -1.0), CostModel(allow_short=True), 1000)
    assert r.units.iloc[-1] == -10
    assert r.final_equity == pytest.approx(1200)


def test_flip_closes_and_reopens_as_two_trades():
    d = bars([100, 100, 100, 100, 100, 100])
    r = backtest(d, sig(d, [1, 1, -1, -1, 0, 0]), CostModel(), 1000)
    assert list(r.trades["direction"]) == ["long", "short"]
    assert list(r.units) == [0, 10, 10, -10, -10, 0]


def test_borrow_cost_charged_on_short():
    d = bars([100] * 300)
    r = backtest(d, sig(d, -1.0), CostModel(annual_borrow_rate=0.10), 1000)
    years = (d.index[-1] - d.index[1]).days / 365.25
    assert r.stats["financing_paid"] == pytest.approx(1000 * 0.10 * years, rel=1e-6)
    assert r.final_equity == pytest.approx(1000 - r.stats["financing_paid"])


def test_drawdown_kill_switch_stops_trading():
    closes = [100, 100, 100, 80, 60, 60, 120, 240]
    d = bars(closes, closes)
    r = backtest(d, sig(d, 1.0), CostModel(), 1000, max_drawdown_stop=0.15)
    assert r.stats["halted"] and r.stats["halt_time"] == d.index[3]
    assert r.units.iloc[4] == 0                       # liquidated at next open
    assert r.equity.iloc[-1] == r.equity.iloc[4] == pytest.approx(600)
    assert len(r.trades) == 1


def test_open_trade_reported_not_counted():
    d = bars([100, 100, 110])
    r = backtest(d, sig(d, 1.0), CostModel(), 1000)
    assert r.trades.empty and r.stats["open_position_at_end"]


def test_fractional_target_and_resize_is_one_trade():
    d = bars([100] * 6)
    r = backtest(d, sig(d, [0.5, 1, 0.5, 0, 0, 0]), CostModel(), 1000)
    assert list(r.units) == [0, 5, 10, 5, 0, 0]
    assert len(r.trades) == 1


def test_leverage_cap_clips_signal():
    d = bars([100] * 4)
    r = backtest(d, sig(d, 5.0), CostModel(max_leverage=2), 1000)
    assert r.units.iloc[-1] == 20


@pytest.mark.parametrize("kwargs", [
    {"initial_capital": 0}, {"max_drawdown_stop": 1.5}])
def test_invalid_arguments(kwargs):
    d = bars([100, 100, 100])
    with pytest.raises(ValueError):
        backtest(d, sig(d, 1.0), CostModel(), **kwargs)


def test_unsorted_data_rejected():
    d = bars([100, 100, 100]).iloc[::-1]
    with pytest.raises(ValueError):
        backtest(d, sig(d, 1.0))


def test_invalid_cost_model():
    with pytest.raises(ValueError):
        CostModel(commission_rate=-0.1)


def test_matches_independent_vectorised_calculation():
    """Zero-cost long/flat result must equal the product of open-to-open
    returns over the bars where the previous bar's signal was 1."""
    d = synthetic_ohlcv(600, seed=9)
    rng = np.random.default_rng(1)
    s = sig(d, np.repeat(rng.integers(0, 2, 60), 10).astype(float))
    r = backtest(d, s, CostModel(), 1000)
    held = s.shift(1).fillna(0).to_numpy()[:-1]           # in position from open i
    open_ret = (d["open"].shift(-1) / d["open"]).to_numpy()[:-1]
    expected = 1000 * np.prod(np.where(held == 1, open_ret, 1.0))
    last_leg = d["close"].iloc[-1] / d["open"].iloc[-1] if s.iloc[-2] == 1 else 1.0
    assert r.final_equity == pytest.approx(expected * last_leg)
