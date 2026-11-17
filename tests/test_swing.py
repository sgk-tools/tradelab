import numpy as np
import pandas as pd
import pytest

from tradelab.swing import (SIGNALS, SwingConfig, _simulate, add_swing, atr,
                            signal_frame, signal_stats, trade_plan)


def bars(close, volume=1000.0, spread=0.5, start="2022-01-03"):
    c = np.asarray(close, float)
    idx = pd.bdate_range(start, periods=len(c))
    v = np.broadcast_to(np.asarray(volume, float), c.shape).copy()
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c,
                         "volume": v}, index=idx)


def uptrend(n=400, daily=0.003, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(daily + rng.normal(0, 0.01, n)))


def test_signals_use_no_future_data():
    d = bars(uptrend(450), volume=np.random.default_rng(2).uniform(500, 2000, 450))
    full = signal_frame(d)
    for cut in (300, 360, 449):
        part = signal_frame(d.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_trend_template_true_in_steady_uptrend_false_in_downtrend():
    assert signal_frame(bars(uptrend(400))).iloc[-1]["TT"]
    assert not signal_frame(bars(uptrend(400, daily=-0.003))).iloc[-1]["TT"]
    assert not signal_frame(bars(uptrend(200))).iloc[-1]["TT"]      # not enough history


def test_breakout_needs_new_high_and_volume():
    c = np.r_[np.full(60, 100.0), 103.0]
    quiet = bars(c, volume=1000.0)
    loud = bars(c, volume=np.r_[np.full(60, 1000.0), 2000.0])
    assert not signal_frame(quiet).iloc[-1]["BO"]
    assert signal_frame(loud).iloc[-1]["BO"]
    assert not signal_frame(bars(np.r_[np.full(60, 100.0), 100.2],
                                 volume=np.r_[np.full(60, 1000.0), 2000.0])).iloc[-1]["BO"]


def test_pocket_pivot_beats_down_day_volume():
    c = np.r_[np.linspace(90, 110, 60), 109, 110, 109, 110, 111]
    vol = np.r_[np.full(60, 1000.0), 1500, 900, 1500, 900, 1600]
    assert signal_frame(bars(c, volume=vol)).iloc[-1]["PP"]
    vol[-1] = 1400
    assert not signal_frame(bars(c, volume=vol)).iloc[-1]["PP"]


def test_pullback_rsi2_above_long_average():
    c = np.r_[np.linspace(50, 120, 260), 118, 115, 112]
    assert signal_frame(bars(c)).iloc[-1]["PB"]
    assert not signal_frame(bars(c[::-1])).iloc[-1]["PB"]


def test_vcp_tight_range_near_high():
    rng = np.random.default_rng(3)
    wide = 100 + np.cumsum(rng.normal(0, 2, 300))
    wide = wide - wide.min() + 50
    c = np.r_[wide, np.full(10, wide.max() * 0.97)]
    d = bars(c, spread=0.5)
    d.iloc[:300, d.columns.get_loc("high")] += 3
    d.iloc[:300, d.columns.get_loc("low")] -= 3
    assert signal_frame(d).iloc[-1]["VCP"]


def test_trade_plan_levels():
    d = bars(np.full(30, 100.0), spread=1.0)            # true range 2 every bar
    plan = trade_plan(d)                                 # O'Neil: -7% / +20%
    assert plan["atr"] == pytest.approx(2.0)
    assert plan["stop"] == pytest.approx(93) and plan["target"] == pytest.approx(120)
    assert plan["risk_pct"] == pytest.approx(0.07)
    assert SwingConfig().target_r == pytest.approx(20 / 7) and SwingConfig().max_hold == 60


def test_simulate_stop_target_and_time_exit():
    cfg = SwingConfig(stop_pct=0.05, target_pct=0.10, max_hold=3)
    n = 10
    o = np.full(n, 100.0); h = o + 1; lo = o - 1; c = o.copy()
    risk = np.full(n, 0.05)                              # stop 95, target 110
    h[2] = 111                                           # target hit on day after entry
    assert _simulate(o, h, lo, c, [0], risk, cfg, 0.0)[0] == pytest.approx(2.0)
    h[2] = 101; lo[2] = 94; h[3] = 120                   # stop hit first
    assert _simulate(o, h, lo, c, [0], risk, cfg, 0.0)[0] == pytest.approx(-1.0)
    lo[2] = 99; h[3] = 101; c[3] = 102.5                 # neither: exit at close of 3rd bar
    assert _simulate(o, h, lo, c, [0], risk, cfg, 0.0)[0] == pytest.approx(0.5)
    o2 = o.copy(); o2[2] = 90; lo2 = lo.copy(); lo2[2] = 89   # gap below the stop
    assert _simulate(o2, h, lo2, c, [0], risk, cfg, 0.0)[0] == pytest.approx(-2.0)
    assert _simulate(o, h, lo, c, [0], risk, cfg, 0.01)[0] == pytest.approx(0.5 - 0.2)
    assert len(_simulate(o, h, lo, c, [8], risk, cfg, 0.0)) == 0   # still open at the end


def test_signal_stats_shape():
    rng = np.random.default_rng(5)
    pm = {f"S{i}": bars(uptrend(500, seed=i), volume=rng.uniform(500, 2000, 500)) for i in range(3)}
    st = signal_stats(pm)
    assert set(st["rules"]) == {"BASE", "TT", *SIGNALS, "TT+SIG"}
    assert st["rules"]["BASE"]["n"] > 0 and st["symbols"] == 3
    assert st["from"] < st["to"]


def test_add_swing_scores_and_sorts():
    table = pd.DataFrame({"symbol": ["UP", "DOWN"], "N": [True, False], "S": [True, True],
                          "L": [True, False], "DOW": [True, False], "rs_rank": [99, 1]})
    prices = {"UP": bars(uptrend(400)), "DOWN": bars(uptrend(400, daily=-0.003))}
    t = add_swing(table.iloc[::-1], prices.get, market_pass=True, use_fundamentals=False)
    assert list(t["symbol"]) == ["UP", "DOWN"] and t["swing_max"].iloc[0] == 6
    up = t.iloc[0]
    assert up["TT"] and up["swing_score"] == 5 + (up["n_signals"] > 0)
    broken = add_swing(table, lambda s: (_ for _ in ()).throw(OSError()), True, False)
    assert not broken["TT"].any() and (broken["swing_score"] <= 4).all()


def test_atr_basic():
    assert atr(bars(np.full(20, 10.0), spread=0.5), 5).iloc[-1] == pytest.approx(1.0)
