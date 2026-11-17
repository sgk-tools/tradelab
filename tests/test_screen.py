import numpy as np
import pandas as pd
import pytest

from tradelab.cli import main
from tradelab.fundamentals import parse_timeseries
from tradelab.report import write_screen_html
from tradelab.screen import (ScreenConfig, annual_eps_growth, below_52w_high,
                             load_universe, market_status, quarterly_yoy_growth,
                             return_on_equity, rs_raw, run_screen, score,
                             trend_state, updown_volume_ratio)
from tradelab.strategies import dow_theory, swing_points


def zigzag(points, step=6):
    """Bars moving linearly between the given turning-point prices."""
    path = [points[0]]
    for a, b in zip(points, points[1:]):
        path += list(np.linspace(a, b, step + 1)[1:])
    c = np.array(path, float)
    idx = pd.bdate_range("2023-01-02", periods=len(c))
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": 1000.0}, index=idx)


def q(values, end="2026-06-30"):
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values), freq="QE"))


def y(values, end="2025-12-31"):
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values), freq="YE"))


# ---- Dow theory -------------------------------------------------------------

def test_swing_points_found_at_turning_points_with_delay():
    d = zigzag([100, 110, 104, 116])
    sp = swing_points(d, swing=3)
    highs, lows = sp["swing_high"].dropna(), sp["swing_low"].dropna()
    assert highs.iloc[0] == 110.5 and lows.iloc[0] == 103.5
    assert highs.index[0] == d.index[6 + 3]          # peak bar 6, known 3 bars later


def test_uptrend_needs_higher_low_and_break_of_high():
    d = zigzag([100, 110, 104, 116, 109, 124])
    t = dow_theory(d, swing=3)
    assert t.iloc[:18].eq(0).all()                   # only one swing low so far
    assert t.iloc[-1] == 1
    first = t[t == 1].index[0]
    assert d["close"][first] > 116.5                 # broke the prior swing high


def test_uptrend_ends_on_close_below_last_swing_low():
    d = zigzag([100, 110, 104, 116, 109, 124, 100])
    t = dow_theory(d, swing=3)
    assert t.max() == 1 and t.iloc[-1] == 0
    end = t[(t == 0) & (t.shift(1) == 1)].index[0]
    assert d["close"][end] < 108.5


def test_downtrend_mirror_and_long_only_clipping():
    d = zigzag([124, 109, 116, 104, 110, 95])
    assert dow_theory(d, swing=3, allow_short=True).iloc[-1] == -1
    assert dow_theory(d, swing=3).eq(0).all()
    assert trend_state(d, 3)[0] == -1


def test_sideways_market_has_no_trend():
    d = zigzag([100, 110, 100, 110, 100, 110, 100])
    assert dow_theory(d, swing=3, allow_short=True).eq(0).all()


# ---- fundamentals rules -------------------------------------------------------

ASOF = pd.Timestamp("2026-08-15")


def test_quarterly_growth_basic():
    assert quarterly_yoy_growth(q([1.0, 1.1, 1.2, 1.3, 1.5]), ASOF) == pytest.approx(0.5)


def test_quarterly_growth_undefined_cases():
    assert np.isnan(quarterly_yoy_growth(q([-1.0, 1, 1, 1, 2.0]), ASOF))     # from a loss
    assert np.isnan(quarterly_yoy_growth(q([1.0, 1.1, 1.2]), ASOF))          # no year-ago
    assert np.isnan(quarterly_yoy_growth(q([1.0, 1, 1, 1, 2.0]), pd.Timestamp("2027-06-01")))
    assert np.isnan(quarterly_yoy_growth(pd.Series(dtype=float), ASOF))


def test_quarterly_growth_matches_by_date_when_quarters_missing():
    s = q([1.0, 9.0, 9.0, 9.0, 2.0]).drop(pd.Timestamp("2025-12-31"))
    assert quarterly_yoy_growth(s, ASOF) == pytest.approx(1.0)
    gap = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-03-31", "2026-06-30"]))
    assert np.isnan(quarterly_yoy_growth(gap, ASOF))


def test_annual_growth():
    cagr, rising = annual_eps_growth(y([1.0, 1.5, 2.0, 8.0]))
    assert cagr == pytest.approx(1.0) and rising
    assert annual_eps_growth(y([1.0, 3.0, 2.0, 8.0])) == (pytest.approx(1.0), False)
    assert np.isnan(annual_eps_growth(y([-1.0, 1.0, 2.0]))[0])
    assert np.isnan(annual_eps_growth(y([1.0, 2.0]))[0])


def test_roe():
    assert return_on_equity(y([10.0, 20.0]), y([100.0, 80.0])) == pytest.approx(0.25)
    assert np.isnan(return_on_equity(y([20.0]), y([-5.0])))
    assert np.isnan(return_on_equity(y([20.0]), y([80.0], end="2024-12-31")))


def test_parse_timeseries_handles_nulls_and_missing_fields():
    payload = {"timeseries": {"result": [
        {"meta": {"type": ["quarterlyDilutedEPS"]}, "quarterlyDilutedEPS": [
            {"asOfDate": "2026-03-31", "reportedValue": {"raw": 1.5}}, None,
            {"asOfDate": "2025-12-31", "reportedValue": {"raw": 1.2}}]},
        {"meta": {"type": ["annualNetIncome"]}}]}}
    f = parse_timeseries(payload)
    assert f["quarterly_eps"].tolist() == [1.2, 1.5]
    assert f["annual_net_income"].empty and f["annual_eps"].empty
    assert all(v.empty for v in parse_timeseries({}).values())


# ---- price rules ----------------------------------------------------------------

def trending(n=300, daily=0.002, vol_up=2000.0, vol_down=1000.0):
    """Rising price with pullbacks: every 12 bars, 4 down bars."""
    steps = np.where(np.arange(n) % 12 < 8, daily * 2.5, -daily * 2)
    c = 100 * np.cumprod(1 + steps)
    idx = pd.bdate_range("2025-06-02", periods=n)
    return pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c,
                         "volume": np.where(steps > 0, vol_up, vol_down)}, index=idx)


def test_price_measurements():
    d = trending()
    assert 0 <= below_52w_high(d) < 0.02
    assert updown_volume_ratio(d) > 3
    assert rs_raw(d) > 0 and rs_raw(trending(daily=-0.002)) < 0
    assert trend_state(d, 3)[0] == 1


def test_market_status_requires_trend_and_average():
    assert market_status(trending(), 3)["pass"]
    m = market_status(trending(daily=-0.002), 3)
    assert not m["pass"] and not m["above_sma200"]


# ---- scoring and full run ---------------------------------------------------------

GOOD = {"quarterly_eps": q([1.0, 1.1, 1.2, 1.3, 1.5]), "quarterly_revenue": q([10, 11, 12, 13, 14.0]),
        "annual_eps": y([1.0, 1.5, 2.2, 3.3]), "annual_net_income": y([5.0, 30.0]),
        "annual_equity": y([90.0, 100.0])}
WEAK = {**GOOD, "quarterly_eps": q([1.0, 1.1, 1.2, 1.3, 1.05])}
PRICES = {"^M": trending(), "WIN": trending(daily=0.004), "MID": trending(daily=0.002),
          "LOSE": trending(daily=-0.002), "NEW": trending(n=100),
          "A1": trending(daily=0.001), "A2": trending(daily=0.0005), "A3": trending(daily=-0.001)}
FUND = {"WIN": GOOD, "MID": WEAK, "LOSE": GOOD, "A1": WEAK, "A2": WEAK, "A3": WEAK}


def run(market="^M", **kw):
    uni = pd.DataFrame({"symbol": ["WIN", "MID", "LOSE", "NEW", "A1", "A2", "A3", "GONE"],
                        "name": "n", "sector": "s"})
    return run_screen(uni, ScreenConfig(dow_swing=3), {"idx": market},
                      price_loader=lambda s: PRICES[s], fundamentals_loader=lambda s: FUND[s], **kw)


def test_full_screen_ranks_and_flags_candidates():
    res = run()
    t = res.table.set_index("symbol")
    assert res.market_pass and res.table["symbol"].iloc[0] == "WIN"
    assert t.loc["WIN", "candidate"] and t.loc["WIN", "score"] == 6
    assert not t.loc["MID", "C"] and not t.loc["MID", "candidate"]
    assert not t.loc["LOSE", ["N", "L", "DOW"]].any() and t.loc["LOSE", "C"]
    assert t["rs_rank"].between(1, 99).all() and t.loc["WIN", "rs_rank"] == 99
    assert set(res.failures["symbol"]) == {"NEW", "GONE"}
    assert "価格データ不足" in res.failures.set_index("symbol").loc["NEW", "reason"]


def test_no_candidates_when_market_fails():
    res = run(market="LOSE")
    assert not res.market_pass and not res.table["candidate"].any()
    assert res.table.set_index("symbol").loc["WIN", "score"] == 6


def test_technical_only_mode_ignores_fundamentals():
    res = run(use_fundamentals=False)
    t = res.table.set_index("symbol")
    assert t.loc["WIN", "max_score"] == 4 and t.loc["WIN", "candidate"]
    assert np.isnan(t.loc["WIN", "eps_growth_q"])


def test_missing_fundamentals_fail_c_and_a_not_crash():
    empty = parse_timeseries({})
    res = run_screen(pd.DataFrame({"symbol": ["WIN"], "name": "", "sector": ""}),
                     ScreenConfig(dow_swing=3), {"idx": "^M"},
                     price_loader=lambda s: PRICES[s], fundamentals_loader=lambda s: empty)
    row = res.table.iloc[0]
    assert not row["C"] and not row["A"] and row["score"] == 4


def test_score_thresholds_are_configurable():
    res = run()
    strict = score(res.table, ScreenConfig(c_min_eps_growth=0.9), True)
    assert not strict["C"].any()


def test_stale_symbol_reported():
    old = trending().copy()
    old.index = old.index - pd.Timedelta(days=60)
    prices = {**PRICES, "OLD": old}
    res = run_screen(pd.DataFrame({"symbol": ["WIN", "OLD"], "name": "", "sector": ""}),
                     ScreenConfig(dow_swing=3), {"idx": "^M"},
                     price_loader=lambda s: prices[s], fundamentals_loader=lambda s: GOOD)
    assert "価格が古い" in res.failures["reason"].iloc[0]


def test_universe_file_and_report(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("aapl\n# comment\nMSFT  # note\n\naapl\n")
    assert load_universe(file=str(f))["symbol"].tolist() == ["AAPL", "MSFT"]
    assert len(load_universe("crypto")) >= 10
    with pytest.raises(ValueError):
        load_universe("nope")
    out = write_screen_html(tmp_path / "s.html", run())
    text = out.read_text(encoding="utf-8")
    assert "WIN" in text and "class='cand'" in text and "未評価" in text
    bad = write_screen_html(tmp_path / "b.html", run(market="LOSE")).read_text(encoding="utf-8")
    assert "買い候補ではない" in bad and "class='cand'" not in bad


def test_cli_rejects_unknown_setting(capsys):
    assert main(["screen", "--set", "bogus=1"]) == 2
    assert "unknown setting" in capsys.readouterr().err


def test_parse_watch_cleans_and_limits():
    from tradelab.screen import parse_watch
    text = "nvda\n7203.t # toyota\n\nNVDA\nbad symbol!\nBTC-USD\nUSDJPY=X\n^GSPC\n"
    assert parse_watch(text) == ["NVDA", "7203.T", "BTC-USD", "USDJPY=X", "^GSPC"]
    assert parse_watch("\n".join(f"S{i}" for i in range(80))) == [f"S{i}" for i in range(50)]


def test_rs_rank_against_reference_universe():
    t = pd.DataFrame({"rs_raw": [0.5, -0.5, np.nan], "eps_growth_q": np.nan, "eps_cagr": np.nan,
                      "eps_rising": False, "roe": np.nan, "below_high": 0.0,
                      "updown_volume": 1.0, "dow_trend": 1})
    ref = pd.Series(np.linspace(-1, 1, 101))
    out = score(t, ScreenConfig(), True, False, rs_reference=ref).sort_index()
    assert out.loc[0, "rs_rank"] == pytest.approx(round(76 / 101 * 98 + 1))
    assert out.loc[1, "rs_rank"] < 30 and np.isnan(out.loc[2, "rs_rank"])
    assert bool(out.loc[0, "L"]) is False or out.loc[0, "rs_rank"] >= 80


def test_parse_jpx_list_keeps_domestic_prime_only():
    from tradelab.screen import parse_jpx_list
    raw = pd.DataFrame({"日付": [20261031] * 5,
                        "コード": [7203, "130A", 1306, 9984, 4385],
                        "銘柄名": ["トヨタ自動車", "新興A", "TOPIX ETF", "ソフトバンクグループ", "メルカリ"],
                        "市場・商品区分": ["プライム（内国株式）", "グロース（内国株式）", "ETF・ETN",
                                     "プライム（内国株式）", "プライム（外国株式）"],
                        "33業種区分": ["輸送用機器", "情報・通信業", "-", "情報・通信業", "情報・通信業"]})
    t = parse_jpx_list(raw)
    assert list(t["symbol"]) == ["7203.T", "9984.T"]
    assert list(t["sector"]) == ["輸送用機器", "情報・通信業"] and t["name"][0] == "トヨタ自動車"
    with pytest.raises(ValueError):
        parse_jpx_list(raw.drop(columns=["コード"]))


def test_etf_universe_and_unknown_name():
    t = load_universe("etf")
    assert {"SPY", "1306.T", "GLD"} <= set(t["symbol"]) and t["symbol"].is_unique
    with pytest.raises(ValueError):
        load_universe("nasdaq")
