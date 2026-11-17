import json

import pandas as pd

from tradelab.app import build_app_data, load_previous, render_app
from tradelab.screen import ScreenConfig, run_screen

from test_screen import FUND, GOOD, PRICES


def screen(symbols, fundamentals=True, market="^M"):
    uni = pd.DataFrame({"symbol": symbols, "name": "N </script> & co", "sector": "s"})
    return run_screen(uni, ScreenConfig(dow_swing=3), {"idx": market},
                      use_fundamentals=fundamentals, price_loader=lambda s: PRICES[s],
                      fundamentals_loader=lambda s: FUND.get(s, GOOD))


def build(previous=None, **kw):
    results = {"us": ("米国株", screen(["WIN", "MID", "LOSE", "A1", "A2", "A3"])),
               "crypto": ("暗号資産", screen(["WIN", "LOSE"], fundamentals=False))}
    return build_app_data(results, lambda s: PRICES[s], lambda s: FUND.get(s, GOOD),
                          previous=previous, **kw)


def test_data_structure_and_json_safety():
    data = build()
    text = json.dumps(data, allow_nan=False)           # raises on NaN/inf
    us = data["lists"]["us"]
    assert [r["s"] for r in us["rows"]][0] == "WIN" and us["crit"][-1] == "TT"
    win = us["rows"][0]
    assert win["p"][:6] == "111111" and len(win["g"]) == len(us["sigs"]) == 4
    assert us["max"] == 8 and win["sc"] == win["p"].count("1") + ("1" in win["g"])
    assert win["cand"] == (win["p"] == "1111111" and "1" in win["g"])
    assert win["st"] < win["cl"] < win["tg"] and 0 < win["rp"] < 1
    assert data["lists"]["crypto"]["crit"] == ["N", "S", "L", "DOW", "TT"]
    assert data["lists"]["crypto"]["max"] == 6 and data["config"]["stop_pct"] == 0.07
    ch = data["charts"]["WIN"]
    assert len(ch["c"]) == len(ch["d"]) <= 130 and ch["d"][-1] == PRICES["WIN"].index[-1].strftime("%Y-%m-%d")
    assert ch["hi"] >= max(ch["c"]) and len(ch["q"]) == 5 and len(ch["a"]) == 4
    assert "generated" in data and "NaN" not in text
    op = data["opens"]["us"]
    assert len(op["d"]) == 60 and op["d"][-1] == PRICES["WIN"].index[-1].strftime("%Y-%m-%d")
    assert op["o"]["WIN"][-1] == float(f'{PRICES["WIN"]["open"].iloc[-1]:.6g}')
    assert set(op["o"]) == set(op["h"]) == set(op["l"]) == {r["s"] for r in us["rows"]}
    assert op["h"]["WIN"][-1] >= op["o"]["WIN"][-1] >= op["l"]["WIN"][-1]


def test_missing_values_become_null_not_nan():
    data = build_app_data({"us": ("米国株", screen(["WIN"]))}, lambda s: PRICES[s],
                          lambda s: {k: v.iloc[:0] for k, v in GOOD.items()})
    assert data["charts"]["WIN"]["q"] == []
    res = run_screen(pd.DataFrame({"symbol": ["WIN"], "name": "", "sector": ""}),
                     ScreenConfig(dow_swing=3), {"idx": "^M"}, price_loader=lambda s: PRICES[s],
                     fundamentals_loader=lambda s: {k: v.iloc[:0] for k, v in GOOD.items()})
    row = build_app_data({"us": ("x", res)}, lambda s: PRICES[s], lambda s: GOOD)["lists"]["us"]["rows"][0]
    assert row["eg"] is None and row["roe"] is None


def test_previous_rank_only_from_a_different_day():
    first = build()
    assert not first["lists"]["us"]["hasPrev"]
    same_day = build(previous=first)
    assert not same_day["lists"]["us"]["hasPrev"]
    older = json.loads(json.dumps(first))
    older["lists"]["us"]["asof"] = "2020-01-01"
    older["lists"]["us"]["rows"] = [r for r in reversed(older["lists"]["us"]["rows"]) if r["s"] != "MID"]
    nxt = build(previous=older)
    rows = {r["s"]: r for r in nxt["lists"]["us"]["rows"]}
    assert nxt["lists"]["us"]["hasPrev"] and rows["MID"]["prev"] is None
    assert rows["WIN"]["prev"] == 5
    again = build(previous=nxt)                         # rebuilt on the same day
    assert {r["s"]: r["prev"] for r in again["lists"]["us"]["rows"]}["WIN"] == 5


def test_chart_failure_does_not_drop_the_row():
    def loader(s):
        raise OSError("cache missing")
    data = build_app_data({"us": ("米国株", screen(["WIN", "MID"]))}, loader, loader)
    assert data["charts"] == {} and len(data["lists"]["us"]["rows"]) == 2


def test_render_writes_embeddable_and_standalone_pages(tmp_path):
    paths = render_app(build(), tmp_path)
    page = paths["page"].read_text(encoding="utf-8")
    index = paths["index"].read_text(encoding="utf-8")
    assert "__DATA__" not in page and page.lstrip().startswith("<title>")
    assert "<!doctype" not in page.lower() and index.lower().startswith("<!doctype html>")
    assert "viewport-fit=cover" in index and page in index
    # a name containing </script> must not terminate the data block early
    block = page.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(block)["lists"]["us"]["rows"][0]["n"] == "N </script> & co"
    assert load_previous(tmp_path)["lists"]["us"]["rows"][0]["s"] == "WIN"
    assert load_previous(tmp_path / "nope") is None


def test_missing_earnings_marked_as_not_measurable_and_bars_split(tmp_path):
    res = run_screen(pd.DataFrame({"symbol": ["WIN"], "name": "", "sector": ""}),
                     ScreenConfig(dow_swing=3), {"idx": "^M"}, price_loader=lambda s: PRICES[s],
                     fundamentals_loader=lambda s: {k: v.iloc[:0] for k, v in GOOD.items()})
    data = build_app_data({"us": ("x", res)}, lambda s: PRICES[s], lambda s: GOOD)
    row = data["lists"]["us"]["rows"][0]
    assert row["p"][:2] == "--" and row["p"][2:6] != "----"
    paths = render_app(data, tmp_path / "site", state_dir=tmp_path / "state")
    page = paths["index"].read_text(encoding="utf-8")
    bars = json.loads((tmp_path / "site" / "bars_us.json").read_text())
    assert bars["d"] and "WIN" in bars["o"] and '"barsFiles":{"us":"bars_us.json"}' in page
    state = json.loads((tmp_path / "state" / "data.json").read_text())
    assert state["lists"]["us"]["rows"][0]["s"] == "WIN" and "opens" not in state
