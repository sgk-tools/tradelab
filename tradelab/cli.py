"""Command line interface: python -m tradelab <command> ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .costs import PRESETS, get_cost_model
from .data import DataError, fetch_ohlcv, load_csv, synthetic_ohlcv
from .engine import backtest
from .metrics import compute_metrics
from .report import fmt, metrics_table, write_html, write_screen_html
from .strategies import STRATEGIES, make_signal
from .validate import walk_forward


def _value(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _pairs(items: list[str] | None, multi: bool) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"error: expected key=value, got '{item}'")
        key, raw = item.split("=", 1)
        out[key] = [_value(v) for v in raw.split(",")] if multi else _value(raw)
    return out


def _load(args) -> tuple[pd.DataFrame, str]:
    if args.csv:
        return load_csv(args.csv), Path(args.csv).name
    return synthetic_ohlcv(n=1500, seed=args.seed), f"synthetic(seed={args.seed})"


def _benchmark(data, cost, capital):
    res = backtest(data, make_signal("buy_and_hold", data), cost, capital)
    return res, compute_metrics(res.equity, res.trades, capital, res.units)


def _notes(args, res_stats: dict | None = None) -> list[str]:
    notes = []
    if not args.csv:
        notes.append("合成データ（ランダムウォーク）による動作確認。この結果に投資判断上の意味はない。")
    if res_stats and res_stats.get("skipped_orders"):
        notes.append(f"資金不足で発注できなかった回数: {res_stats['skipped_orders']}。"
                     "最小売買単位に対して資金が不足している。")
    if res_stats and res_stats.get("halted"):
        notes.append(f"損失上限に到達し {res_stats['halt_time']} に売買を停止した。")
    return notes


def cmd_backtest(args) -> int:
    data, label = _load(args)
    cost = get_cost_model(args.market)
    params = _pairs(args.params, multi=False)
    res = backtest(data, make_signal(args.strategy, data, **params), cost,
                   args.capital, args.max_dd)
    m = compute_metrics(res.equity, res.trades, args.capital, res.units)
    bench, bm = _benchmark(data, cost, args.capital)
    print(f"data: {label}  bars: {len(data)}  market: {args.market}  "
          f"strategy: {args.strategy} {params}")
    print(metrics_table({args.strategy: m, "buy_and_hold": bm}).to_string())
    print(f"total cost paid: {fmt(res.stats['total_cost'], 'money')}")
    for note in _notes(args, res.stats):
        print("NOTE:", note)
    if args.out:
        path = write_html(
            args.out, f"バックテスト: {args.strategy}",
            f"データ: {label} / 市場設定: {args.market} / パラメータ: {params} / "
            f"初期資金: {args.capital:,.0f}",
            {args.strategy: res.equity, "buy_and_hold": bench.equity},
            {args.strategy: m, "buy_and_hold": bm}, _notes(args, res.stats))
        print("report:", path.resolve())
    return 0


def cmd_walkforward(args) -> int:
    data, label = _load(args)
    cost = get_cost_model(args.market)
    grid = _pairs(args.grid, multi=True)
    wf = walk_forward(data, args.strategy, grid, cost, args.capital,
                      args.train, args.test, args.metric)
    m = compute_metrics(wf.equity, wf.trades, args.capital, wf.units)
    oos = data.loc[wf.equity.index[0]:wf.equity.index[-1]]
    bench, bm = _benchmark(oos, cost, args.capital)
    folds = wf.folds.copy()
    for col in ("train_start", "train_end", "test_start", "test_end"):
        folds[col] = folds[col].dt.strftime("%Y-%m-%d")
    folds["params"] = folds["params"].astype(str)
    folds["in_sample_score"] = folds["in_sample_score"].map(lambda v: fmt(v, "num"))
    folds["oos_score"] = folds["oos_score"].map(lambda v: fmt(v, "num"))
    folds["oos_return"] = folds["oos_return"].map(lambda v: fmt(v, "pct"))
    print(f"data: {label}  bars: {len(data)}  market: {args.market}  "
          f"strategy: {args.strategy}  combinations tried: {wf.n_combinations}")
    print(metrics_table({"walk-forward (OOS)": m, "buy_and_hold": bm}).to_string())
    print(folds.drop(columns=["train_start", "train_end"]).to_string(index=False))
    if args.out:
        path = write_html(
            args.out, f"ウォークフォワード検証: {args.strategy}",
            f"データ: {label} / 市場設定: {args.market} / 探索グリッド: {grid} "
            f"({wf.n_combinations}通り) / 学習 {args.train} バー・検証 {args.test} バー",
            {"walk-forward (OOS)": wf.equity, "buy_and_hold": bench.equity},
            {"walk-forward (OOS)": m, "buy_and_hold": bm}, _notes(args),
            {"フォールド別の結果（学習期間で選んだパラメータを未使用期間で評価）": folds})
        print("report:", path.resolve())
    return 0


def cmd_fetch(args) -> int:
    data = fetch_ohlcv(args.symbol, args.source, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out)
    if data.attrs.get("repaired_bars"):
        print(f"NOTE: repaired {data.attrs['repaired_bars']} bars whose high/low "
              "missed open/close by <= 0.5% (vendor data inconsistency)")
    print(f"saved {len(data)} bars ({data.index[0].date()} .. "
          f"{data.index[-1].date()}) to {out.resolve()}")
    return 0


def cmd_screen(args) -> int:
    from dataclasses import fields, replace

    from .screen import ScreenConfig, load_universe, run_screen

    cfg = ScreenConfig()
    overrides = _pairs(args.set, multi=False)
    known = {f.name for f in fields(ScreenConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown setting {sorted(unknown)}. choices: {sorted(known)}")
    cfg = replace(cfg, **overrides)

    universe = load_universe(None if args.file else args.universe, args.file)
    crypto = args.universe == "crypto" and not args.file
    market = {"BTC/USD": "BTC-USD"} if crypto else \
        {"S&P 500": "^GSPC", "ダウ工業株": "^DJI", "ダウ輸送株": "^DJT"}

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    name = Path(args.file).name if args.file else {"sp500": "S&P 500", "crypto": "主要暗号資産"}[args.universe]
    res = run_screen(universe, cfg, market, use_fundamentals=not crypto,
                     cache_dir=args.cache, workers=args.workers,
                     universe_name=name, progress=progress)
    t = res.table
    print(f"universe: {name}  as of: {res.asof.date()}  evaluated: {len(t)}  "
          f"failed: {len(res.failures)}")
    for label, m in res.market.items():
        print(f"market {label}: {'PASS' if m['pass'] else 'FAIL'} "
              f"(dow trend {m['trend']:+d}, above 200d avg: {m['above_sma200']})")
    print(f"candidates (all criteria + market): {int(t['candidate'].sum())}")
    cols = ["symbol", "score"] + ([] if crypto else ["eps_growth_q", "eps_cagr", "roe"]) \
        + ["below_high", "updown_volume", "rs_rank", "dow_trend", "close"]
    print(t[cols].head(args.top).round(2).to_string(index=False))
    if args.out:
        out = Path(args.out)
        print("report:", write_screen_html(out, res, args.top).resolve())
        t.to_csv(out.with_suffix(".csv"), index=False)
        print("csv:", out.with_suffix(".csv").resolve())
    return 0


def cmd_app(args) -> int:
    from datetime import datetime, timedelta, timezone

    from . import ledger as paper
    from .app import build_app_data, load_previous, render_app
    from .screen import (ScreenConfig, _cached_fundamentals, _cached_meta,
                         _cached_prices, fundamentals_cache_dir, load_universe,
                         parse_watch, run_screen)
    from .swing import SwingConfig, signal_stats

    cfg = ScreenConfig()
    today = datetime.now(timezone.utc).date()
    cache = Path(args.cache) / today.isoformat()
    fund_cache = fundamentals_cache_dir(args.cache, today)
    history = 1500                     # longer history for the signal record
    start = (today - timedelta(days=history)).isoformat()
    state = Path(args.state or args.out)
    markets = set(args.markets.split(","))

    def prices(symbol):
        return _cached_prices(symbol, cache, start)

    def progress(done, total):
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    def screen(key, label, universe, market, fundamentals, name):
        print(f"screening {label} ({len(universe)})...", file=sys.stderr)
        res = run_screen(universe, cfg, market, use_fundamentals=fundamentals,
                         cache_dir=args.cache, workers=args.workers, universe_name=name,
                         progress=progress, history_days=history)
        share = len(res.failures) / max(1, len(res.table) + len(res.failures))
        print(f"  {label}: evaluated {len(res.table)} failed {len(res.failures)} ({share:.0%})",
              file=sys.stderr)
        return res, share

    lists, currencies, shares = {}, {}, {}
    if "us" in markets:
        us, shares["us"] = screen("us", "US stocks", load_universe("sp500"),
                                  {"S&P 500": "^GSPC", "ダウ工業株": "^DJI", "ダウ輸送株": "^DJT"},
                                  True, "S&P 500 構成銘柄")
        if shares["us"] > 0.2:
            raise ValueError(f"{shares['us']:.0%} of US symbols could not be evaluated; "
                             "refusing to publish an incomplete ranking")
        lists["us"] = ("米国株", us)
    if "jp" in markets:
        try:
            jp_universe = load_universe("jp_prime", cache_dir=args.universe)
        except Exception as exc:       # keep the other markets publishing
            print(f"WARNING: Japanese stocks skipped: {exc}", file=sys.stderr)
            jp_universe = None
        if jp_universe is not None:
            jp, shares["jp"] = screen("jp", "JP prime", jp_universe,
                                      {"TOPIX（1306.T）": "1306.T", "日経平均": "^N225"},
                                      True, "東証プライム（内国株式）")
            lists["jp"] = ("日本株", jp)
            currencies["jp"] = {s: "JPY" for s in jp.table["symbol"]}
    if "etf" in markets:
        etf, shares["etf"] = screen("etf", "ETF", load_universe("etf"),
                                    {"S&P 500": "^GSPC", "TOPIX（1306.T）": "1306.T"},
                                    False, "主要ETF（米国・日本）")
        lists["etf"] = ("ETF", etf)
        currencies["etf"] = {s: "JPY" if s.endswith(".T") else "USD" for s in etf.table["symbol"]}
    if "crypto" in markets:
        cr, shares["crypto"] = screen("crypto", "crypto", load_universe("crypto"),
                                      {"BTC/USD": "BTC-USD"}, False,
                                      "主要暗号資産 12 銘柄（米ドル建て）")
        lists["crypto"] = ("暗号資産", cr)
    if not lists:
        raise ValueError("no market selected")

    watch = parse_watch(Path(args.watch).read_text(encoding="utf-8")) if args.watch else []
    known = set()
    for _, res in lists.values():
        known |= set(res.table["symbol"]) | set(res.failures["symbol"])
    extra = [s for s in watch if s not in known]
    if extra:
        print(f"screening {len(extra)} watch-list symbols...", file=sys.stderr)
        metas = {}
        for s in extra:
            try:
                metas[s] = _cached_meta(s, cache)
            except Exception as exc:
                print(f"  {s}: no metadata ({type(exc).__name__})", file=sys.stderr)
        uni = pd.DataFrame({"symbol": extra,
                            "name": [metas.get(s, {}).get("name", "") for s in extra],
                            "sector": [metas.get(s, {}).get("exchange", "") for s in extra]})
        ref = lists["us"][1].table["rs_raw"] if "us" in lists else None
        try:
            wl = run_screen(uni, cfg, {"S&P 500": "^GSPC"}, use_fundamentals=False,
                            cache_dir=args.cache, workers=args.workers,
                            universe_name="お気に入り（一覧外の銘柄）",
                            history_days=history, rs_reference=ref)
            lists["watch"] = ("お気に入り", wl)
            currencies["watch"] = {s: metas.get(s, {}).get("currency", "USD") for s in extra}
        except ValueError as exc:      # none of them could be evaluated
            print(f"watch list skipped: {exc}", file=sys.stderr)

    print("replaying swing signals over history...", file=sys.stderr)
    stats = {}
    for key, (_, res) in lists.items():
        if key == "watch":
            continue
        stats[key] = signal_stats({s: prices(s) for s in res.table["symbol"]}, crypto=key == "crypto")
    try:
        fxd = prices("USDJPY=X")
        fx = {"usdjpy": round(float(fxd["close"].iloc[-1]), 2),
              "asof": fxd.index[-1].strftime("%Y-%m-%d"), "rates": {"JPY": 1.0}}
        fx["rates"]["USD"] = fx["usdjpy"]
    except Exception as exc:          # sizing falls back to a manual rate
        print(f"USDJPY unavailable: {exc}", file=sys.stderr)
        fx = None
    for cur in sorted(set((currencies.get("watch") or {}).values()) - {"USD", "JPY"}) if fx else []:
        try:                           # yen rate of any other quote currency
            fx["rates"][cur] = round(float(prices(f"{cur}JPY=X")["close"].iloc[-1]), 4)
        except Exception as exc:
            print(f"{cur}JPY unavailable: {exc}", file=sys.stderr)
    data = build_app_data(
        lists, prices, lambda s: _cached_fundamentals(s, fund_cache),
        previous=load_previous(state), stats=stats, fx=fx, currencies=currencies,
        charts_per_list=args.charts)
    data["failShare"] = {k: round(v, 3) for k, v in shares.items()}
    from . import contest
    ccfg = contest.load_config(args.contest)
    if ccfg.get("enabled") and ccfg.get("csv_url"):
        try:
            symbols = {s: k for k, (_, res) in lists.items() if k != "watch" for s in res.table["symbol"]}
            data["contest"] = contest.build(ccfg, contest.fetch_csv(ccfg["csv_url"]), symbols,
                                            prices, prices("USDJPY=X")["close"])
            print(f"contest: {len(data['contest']['players'])} players, "
                  f"{len(data['contest']['orders'])} orders, unverified {data['contest']['unverified']}")
        except Exception as exc:       # the ranking pages still publish
            print(f"WARNING: contest skipped: {exc}", file=sys.stderr)
            data["contest"] = {"error": str(exc)[:200], "form": ccfg.get("form_url", "")}
    book_path = state / "ledger.json"
    book = paper.load(book_path)
    sc = SwingConfig()
    paper_keys = [k for k in data["lists"] if k != "watch"]
    for key in paper_keys:                    # the watch list is not paper-traded
        L = data["lists"][key]
        paper.add_signals(book, key, L["rows"], L["asof"], L["marketPass"])
    paper.evaluate(book, prices, sc, cost={k: sc.cost_crypto if k == "crypto" else sc.cost_stock
                                          for k in paper_keys})
    data["paper"] = {key: paper.for_page(book, key) for key in paper_keys}
    paths = render_app(data, args.out, state_dir=state)
    paper.save(book, book_path)
    paths["ledger"] = book_path
    for key in paper_keys:
        L = data["lists"][key]
        rows = L["rows"]
        print(f"{key}: as of {L['asof']}  evaluated {len(rows)}  failed {L['failed']}  "
              f"market {'PASS' if L['marketPass'] else 'FAIL'}  "
              f"candidates {sum(r['cand'] for r in rows)}  "
              f"top5 {', '.join(r['s'] for r in rows[:5])}")
        ps = data["paper"][key]["summary"]
        print(f"  paper since {ps['started']}: closed {ps['closed']['n']} "
              f"avgR {ps['closed']['avg']}  open {ps['open']}  pending {ps['pending']}")
        rec = stats[key]["rules"]
        print(f"  record {stats[key]['from']}..{stats[key]['to']}: " + "  ".join(
            f"{k} n={v['n']} avgR={v['avg_r']:+.2f}" for k, v in rec.items() if v["n"]))
    if "watch" in data["lists"]:
        W = data["lists"]["watch"]
        print(f"watch: evaluated {len(W['rows'])} failed {W['failed']}: "
              + ", ".join(r["s"] for r in W["rows"]))
    elif watch:
        print(f"watch: {len(watch)} symbols, none outside the main lists")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradelab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--csv", help="OHLCV CSV file. Omit to use synthetic data.")
        sp.add_argument("--seed", type=int, default=0, help="synthetic data seed")
        sp.add_argument("--market", choices=list(PRESETS), default="crypto",
                        help="cost/constraint preset")
        sp.add_argument("--strategy", choices=list(STRATEGIES), default="sma_cross")
        sp.add_argument("--capital", type=float, default=300_000.0)
        sp.add_argument("--out", help="write an HTML report to this path")

    b = sub.add_parser("backtest", help="run one strategy with fixed parameters")
    common(b)
    b.add_argument("--params", nargs="*", metavar="K=V", help="e.g. fast=20 slow=100")
    b.add_argument("--max-dd", type=float, default=None,
                   help="kill switch: stop trading after this drawdown (0-1)")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="out-of-sample validation over a grid")
    common(w)
    w.add_argument("--grid", nargs="+", required=True, metavar="K=V1,V2",
                   help="e.g. fast=10,20,50 slow=100,200")
    w.add_argument("--train", type=int, default=500, help="training bars per fold")
    w.add_argument("--test", type=int, default=125, help="test bars per fold")
    w.add_argument("--metric", default="sharpe",
                   help="metric to optimise in training (default: sharpe)")
    w.set_defaults(func=cmd_walkforward)

    s = sub.add_parser("screen", help="CAN SLIM + Dow theory stock screener")
    s.add_argument("--universe", choices=["sp500", "crypto"], default="sp500")
    s.add_argument("--file", help="text file with one Yahoo symbol per line "
                                  "(overrides --universe)")
    s.add_argument("--set", nargs="*", metavar="K=V",
                   help="override a threshold, e.g. c_min_eps_growth=0.2 dow_swing=10")
    s.add_argument("--top", type=int, default=50, help="rows to show")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--cache", default="data/cache", help="download cache directory")
    s.add_argument("--out", help="write HTML (and CSV) report to this path")
    s.set_defaults(func=cmd_screen)

    a = sub.add_parser("app", help="build the mobile ranking app (US stocks + crypto)")
    a.add_argument("--out", default="dist", help="output directory")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--cache", default="data/cache", help="download cache directory")
    a.add_argument("--watch", help="watch-list file, one Yahoo symbol per line "
                   "(symbols outside the main lists are added as an extra list)")
    a.add_argument("--state", help="directory for data.json / ledger.json carried between "
                   "runs (default: --out)")
    a.add_argument("--markets", default="us,jp,etf,crypto",
                   help="comma-separated lists to build: us, jp, etf, crypto")
    a.add_argument("--universe", default="data/universe",
                   help="where the downloaded JPX stock list is kept")
    a.add_argument("--charts", type=int, default=40, help="price charts per list (top N)")
    a.add_argument("--contest", default="contest/config.json",
                   help="contest settings (Google Form CSV); disabled when missing")
    a.set_defaults(func=cmd_app)

    f = sub.add_parser("fetch", help="download OHLCV data to CSV")
    f.add_argument("--symbol", required=True)
    f.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
    f.add_argument("--start")
    f.add_argument("--end")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # network errors etc.
        if exc.__class__.__module__.startswith(("requests", "urllib3")):
            print(f"error: download failed: {exc}", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
