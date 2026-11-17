import json

import numpy as np
import pandas as pd
import pytest

from tradelab import ledger as paper
from tradelab.swing import SwingConfig


def prices(path, start="2026-01-05"):
    c = np.asarray(path, float)
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": 1000.0}, index=idx)


def row(sym, g="1000", rp=0.05, sc=7, cand=False):
    return {"s": sym, "g": g, "rp": rp, "sc": sc, "cand": cand}


def test_add_signals_limits_and_skips_busy_symbols():
    book = {"trades": []}
    rows = [row("A"), row("B", g="0000"), row("C", rp=None), row("D"), row("E")]
    assert paper.add_signals(book, "us", rows, "2026-01-05", True, per_day=2) == 2
    assert [t["s"] for t in book["trades"]] == ["A", "D"]
    assert book["trades"][0]["rk"] == 1 and book["trades"][1]["rk"] == 4
    assert paper.add_signals(book, "us", rows, "2026-01-05", True) == 0   # same date: once
    assert paper.add_signals(book, "us", rows, "2026-01-06", True) == 1   # A, D busy; E new
    assert paper.add_signals(book, "us", rows, "2026-01-07", True) == 0   # all busy
    assert paper.add_signals(book, "crypto", rows, "2026-01-06", False) == 3
    assert book["started"] == {"us": "2026-01-05", "crypto": "2026-01-06"}


def test_evaluate_pending_open_and_closed():
    cfg = SwingConfig(stop_pct=0.05, target_pct=0.10, max_hold=5)
    data = {"P": prices([100] * 3),                       # signal on last bar: pending
            "O": prices([100, 100, 101, 102]),            # entered, not exited
            "T": prices([100, 100, 105, 109.6, 111]),     # high 110.1 hits target 110
            "G": prices([100, 100, 105, 112]),            # opens above target: exit at open
            "S": prices([100, 100, 96, 95.3, 94]),        # low 94.8 hits stop 95
            "X": prices([100] * 8)}                       # time exit after 5 bars
    book = {"trades": []}
    for s, d in [("P", "2026-01-07"), ("O", "2026-01-05"), ("T", "2026-01-05"),
                 ("S", "2026-01-05"), ("X", "2026-01-05"), ("G", "2026-01-05")]:
        book["trades"].append({"k": "us", "s": s, "d": d, "rp": 0.05, "m": True, "st": "pending"})
    book["trades"].append({"k": "us", "s": "GONE", "d": "2026-01-05", "rp": 0.05, "m": True, "st": "pending"})
    paper.evaluate(book, lambda s: data[s], cfg, cost={"us": 0.0})
    t = {x["s"]: x for x in book["trades"]}
    assert t["P"]["st"] == "pending"
    assert t["O"]["st"] == "open" and t["O"]["ep"] == 100 and t["O"]["r"] == pytest.approx(0.4)
    assert t["T"]["st"] == "closed" and t["T"]["why"] == "target" and t["T"]["r"] == pytest.approx(2)
    assert t["G"]["why"] == "target" and t["G"]["r"] == pytest.approx(2.4)
    assert t["S"]["why"] == "stop" and t["S"]["r"] == pytest.approx(-1) and t["S"]["xd"] == "2026-01-08"
    assert t["X"]["why"] == "time" and t["X"]["r"] == pytest.approx(0) and t["X"]["bars"] == 5
    assert t["GONE"]["st"] == "pending"                   # missing data keeps state
    s = paper.summary(book, "us")
    assert s["closed"]["n"] == 4 and s["closed"]["sum"] == pytest.approx(3.4)
    assert s["open"] == 1 and s["pending"] == 2
    page = paper.for_page(book, "us")
    assert len(page["live"]) == 3 and page["done"][0]["xd"] >= page["done"][-1]["xd"]


def test_cost_reduces_r_and_roundtrip(tmp_path):
    book = {"trades": [{"k": "us", "s": "T", "d": "2026-01-05", "rp": 0.05, "m": True, "st": "pending"}]}
    paper.evaluate(book, lambda s: prices([100, 100, 105, 109.6]), SwingConfig(stop_pct=0.05, target_pct=0.10),
                   cost={"us": 0.01})
    assert book["trades"][0]["r"] == pytest.approx(2 - 0.2)
    paper.save(book, tmp_path / "l.json")
    assert paper.load(tmp_path / "l.json") == json.loads(json.dumps(book))
    assert paper.load(tmp_path / "missing.json")["trades"] == []
