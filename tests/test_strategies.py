import numpy as np
import pandas as pd
import pytest

from tradelab import STRATEGIES, make_signal, synthetic_ohlcv
from tradelab.strategies import rsi

PARAMS = {
    "buy_and_hold": [{}],
    "sma_cross": [{"fast": 5, "slow": 30}, {"fast": 5, "slow": 30, "allow_short": True}],
    "donchian_breakout": [{"entry": 20, "exit": 10},
                          {"entry": 20, "exit": 10, "allow_short": True}],
    "rsi_reversion": [{"period": 5}, {"period": 5, "allow_short": True}],
    "momentum": [{"lookback": 30}, {"lookback": 30, "allow_short": True}],
    "dow_theory": [{"swing": 3}, {"swing": 3, "allow_short": True}],
}
CASES = [(name, p) for name, plist in PARAMS.items() for p in plist]


def test_every_registered_strategy_is_covered():
    assert set(PARAMS) == set(STRATEGIES)


@pytest.mark.parametrize("name,params", CASES)
def test_no_lookahead(name, params):
    """Signal at bar t must not change when future bars are removed."""
    data = synthetic_ohlcv(400, seed=11)
    full = make_signal(name, data, **params)
    for cut in (60, 150, 287, 399):
        part = make_signal(name, data.iloc[:cut], **params)
        pd.testing.assert_series_equal(part, full.iloc[:cut], check_names=False)


@pytest.mark.parametrize("name,params", CASES)
def test_signal_shape_and_range(name, params):
    data = synthetic_ohlcv(400, seed=12)
    s = make_signal(name, data, **params)
    assert s.index.equals(data.index) and not s.isna().any()
    allowed = {-1.0, 0.0, 1.0} if params.get("allow_short") else {0.0, 1.0}
    assert set(s.unique()) <= allowed


@pytest.mark.parametrize("name,params", [c for c in CASES if c[0] != "buy_and_hold"])
def test_strategy_actually_trades(name, params):
    s = make_signal(name, synthetic_ohlcv(600, seed=13), **params)
    assert s.diff().abs().sum() >= 2


def test_sma_cross_direction():
    idx = pd.bdate_range("2024-01-01", periods=60)
    up = pd.DataFrame({"close": np.linspace(100, 200, 60)}, index=idx)
    s = make_signal("sma_cross", up, fast=3, slow=10)
    assert (s.iloc[:9] == 0).all() and (s.iloc[9:] == 1).all()
    down = pd.DataFrame({"close": np.linspace(200, 100, 60)}, index=idx)
    assert (make_signal("sma_cross", down, fast=3, slow=10) == 0).all()
    assert (make_signal("sma_cross", down, fast=3, slow=10, allow_short=True).iloc[9:] == -1).all()


def test_rsi_bounds_and_extremes():
    idx = pd.bdate_range("2024-01-01", periods=50)
    assert rsi(pd.Series(np.linspace(1, 2, 50), index=idx), 14).iloc[-1] == 100
    assert rsi(pd.Series(np.linspace(2, 1, 50), index=idx), 14).iloc[-1] == pytest.approx(0)
    r = rsi(synthetic_ohlcv(300, seed=5)["close"], 14).dropna()
    assert r.between(0, 100).all()


@pytest.mark.parametrize("name,params", [
    ("sma_cross", {"fast": 50, "slow": 10}),
    ("donchian_breakout", {"entry": 1}),
    ("rsi_reversion", {"lower": 80}),
    ("momentum", {"lookback": 0}),
    ("dow_theory", {"swing": 0}),
    ("nope", {}),
])
def test_invalid_parameters(name, params):
    with pytest.raises(ValueError):
        make_signal(name, synthetic_ohlcv(100), **params)
