from datetime import datetime, timezone

import pandas as pd
import pytest

from tradelab import contest as C

HEAD = "時刻,ニックネーム,銘柄,売買,種類,数量,価格,損切り,利確,取消番号\n"


def bars(opens, start="2026-11-16", highs=None, lows=None, closes=None, freq="B"):
    idx = pd.date_range(start, periods=len(opens), freq=freq)
    o = pd.Series(opens, index=idx, dtype=float)
    return pd.DataFrame({"open": o, "high": highs if highs is not None else o + 1,
                         "low": lows if lows is not None else o - 1,
                         "close": closes if closes is not None else o, "volume": 1000.0}, index=idx)


FX = pd.Series([150.0] * 40, index=pd.date_range("2026-11-01", periods=40))
LISTS = {"AAA": "us", "7203.T": "jp", "BTC-USD": "crypto", "SPY": "etf"}


def run(rows, data, capital=10_000_000, start="2026-11-01"):
    orders, bad = C.load_orders(HEAD + rows, LISTS, start=start)
    players = C.simulate(orders, lambda s: data[s], FX, capital)
    return orders, bad, players


def test_parse_rejects_and_lots():
    rows = ("2026/11/16 20:00:00,,AAA,買い,成行,1,,,,\n"            # wrong passcode -> blank name
            "2026/11/16 20:00:00,ken,ZZZ,買い,成行,1,,,,\n"         # not in lists
            "2026/11/16 20:00:00,ken,7203.T,買い,成行,150,,,,\n"    # 100-share lots
            "2026/11/16 20:00:00,ken,7203.T,買い,成行,50,,,,\n"
            "2026/11/16 20:00:00,ken,AAA,買い,指値,1,,,,\n"         # limit without price
            "2026/10/01 20:00:00,ken,AAA,買い,成行,1,,,,\n"         # before contest start
            "2026/11/16 20:00:00,ken,BTC-USD,買い,成行,0.123456,,,,\n")
    orders, bad = C.load_orders(HEAD + rows, LISTS, start="2026-11-01")
    assert bad == 1
    st = {o.id: (o.status, o.reason, o.qty) for o in orders}
    assert st[2][0] == "rejected" and "対象外" in st[2][1]
    assert st[3] == ("open", "", 100.0)
    assert st[4][0] == "rejected" and "売買単位" in st[4][1]
    assert st[5][0] == "rejected" and "価格" in st[5][1]
    assert st[6][0] == "rejected" and "開始前" in st[6][1]
    assert st[7][2] == pytest.approx(0.1234)


def test_min_fill_date_by_market():
    t = datetime(2026, 11, 17, 1, 0, tzinfo=timezone.utc)        # 10:00 JST, 20:00 NY (16th)
    assert C.min_fill_date(t, "us", "AAA") == "2026-11-17"        # after NY open on the 16th
    assert C.min_fill_date(t, "jp", "7203.T") == "2026-11-18"     # after Tokyo 09:00
    assert C.min_fill_date(t, "crypto", "BTC-USD") == "2026-11-18"
    early = datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc)    # 08:00 JST on the 17th
    assert C.min_fill_date(early, "jp", "7203.T") == "2026-11-17"


def test_market_limit_stop_cancel_and_cash():
    # 16th Mon .. 20th Fri; order times are JST evenings (= NY morning before open? no: 20:00 JST = 06:00 NY)
    data = {"AAA": bars([100, 102, 98, 105, 110], highs=[101, 103, 99, 112, 111],
                        lows=[99, 101, 95, 104, 108])}
    rows = ("2026/11/16 20:00:00,ken,AAA,買い,成行,10,,,,\n"       # 06:00 NY 16th -> fills 16th open 100
            "2026/11/16 20:00:00,amy,AAA,買い,指値,10,96,,,\n"     # low 95 on 18th -> fills at 96
            "2026/11/16 20:00:00,bob,AAA,買い,逆指値,10,111,,,\n"   # high 112 on 19th -> 111
            "2026/11/16 20:00:00,cat,AAA,買い,指値,10,50,,,\n"
            "2026/11/17 20:00:00,cat,,取消,,,,,,4\n"               # cancels id 4 from the 17th
            "2026/11/16 20:00:00,dan,AAA,買い,成行,1000000,,,,\n")  # cannot pay
    orders, _, players = run(rows, data)
    o = {x.id: x for x in orders}
    assert (o[1].fill_date, o[1].fill_px) == ("2026-11-16", 100)
    assert (o[2].fill_date, o[2].fill_px) == ("2026-11-18", 96)
    assert (o[3].fill_date, o[3].fill_px) == ("2026-11-19", 111)
    assert o[4].status == "cancelled"
    assert o[6].status == "rejected" and "資金不足" in o[6].reason
    ken = players["ken"]
    assert ken.cash == pytest.approx(10_000_000 - 10 * 100 * 150 * 1.001)
    assert ken.equity["2026-11-20"] == pytest.approx(ken.cash + 10 * 110 * 150)


def test_auto_stop_and_sell_and_leaderboard():
    data = {"7203.T": bars([1000, 1010, 900, 905, 950], lows=[990, 1000, 880, 900, 940]),
            "AAA": bars([100, 100, 100, 120, 130], highs=[101, 101, 101, 125, 131])}
    rows = ("2026/11/16 08:00:00,ken,7203.T,買い,成行,100,,930,1100,\n"   # Tokyo before 09:00 -> 16th
            "2026/11/16 20:00:00,amy,AAA,買い,成行,100,,,,\n"
            "2026/11/19 08:00:00,amy,AAA,売り,成行,100,,,,\n")              # 18:00 NY on the 18th -> 19th open 120
    orders, _, players = run(rows, data)
    ken, amy = players["ken"], players["amy"]
    assert ken.trades[0]["why"] == "stop" and ken.trades[0]["d"] == "2026-11-18"
    assert ken.trades[0]["p"] == 900                       # opened at 900, already below the 930 stop
    assert amy.trades[0]["p"] == 120 and amy.trades[0]["d"] == "2026-11-19"
    board = C.leaderboard(players, 10_000_000, "2026-11-20")
    names = sorted(board["players"], key=lambda p: -p["ret"]["all"])
    assert names[0]["name"] == "amy" and names[0]["ret"]["all"] > 0 > names[1]["ret"]["all"]
    assert board["starts"] == {"week": "2026-11-16", "month": "2026-11-01", "year": "2026-01-01"}
    assert names[0]["ret"]["week"] == pytest.approx(names[0]["ret"]["all"])


def test_gap_below_stop_sells_at_open():
    data = {"AAA": bars([100, 100, 80, 80], lows=[99, 99, 79, 79])}
    rows = "2026/11/16 20:00:00,ken,AAA,買い,成行,10,,93,,\n"
    _, _, players = run(rows, data)
    assert players["ken"].trades[0]["p"] == 80 and players["ken"].trades[0]["why"] == "stop"
