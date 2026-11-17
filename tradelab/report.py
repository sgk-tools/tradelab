"""Console tables and a self-contained HTML report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

_ROWS = [
    ("final_equity", "最終資産", "money"),
    ("total_return", "総リターン", "pct"),
    ("cagr", "年率リターン (CAGR)", "pct"),
    ("volatility", "年率ボラティリティ", "pct"),
    ("sharpe", "シャープレシオ", "num"),
    ("sortino", "ソルティノレシオ", "num"),
    ("max_drawdown", "最大ドローダウン", "pct"),
    ("max_drawdown_bars", "最長ドローダウン期間 (バー)", "int"),
    ("calmar", "カルマーレシオ", "num"),
    ("exposure", "保有期間の割合", "pct"),
    ("trades", "決済済み取引数", "int"),
    ("win_rate", "勝率", "pct"),
    ("profit_factor", "プロフィットファクター", "num"),
    ("avg_trade_pnl", "1取引あたり平均損益", "money"),
    ("avg_bars_held", "平均保有期間 (バー)", "num"),
]


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if kind == "pct":
        return f"{value * 100:+.2f}%" if np.isfinite(value) else "-"
    if kind == "money":
        return f"{value:,.0f}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.2f}" if np.isfinite(value) else "∞"


def metrics_table(columns: dict[str, dict]) -> pd.DataFrame:
    """Formatted comparison table; ``columns`` maps a label to a metrics dict."""
    data = {name: [fmt(m.get(key), kind) for key, _, kind in _ROWS]
            for name, m in columns.items()}
    return pd.DataFrame(data, index=[label for _, label, _ in _ROWS])


def _chart(curves: dict[str, pd.Series]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    colors = ["#2563eb", "#9ca3af", "#d97706", "#059669"]
    for (name, eq), color in zip(curves.items(), colors):
        ax1.plot(eq.index, eq.values, label=name, color=color, linewidth=1.4)
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=color, linewidth=1.0)
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left", frameon=False)
    ax2.set_ylabel("Drawdown %")
    for ax in (ax1, ax2):
        ax.grid(True, color="#e5e7eb", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


_CSS = """
body{font-family:-apple-system,'Segoe UI','Hiragino Sans','Noto Sans JP',sans-serif;
margin:0;background:#f8fafc;color:#0f172a;line-height:1.6}
main{max-width:980px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.sub{color:#64748b;font-size:13px;margin:0 0 16px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid #eef2f7;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:#475569;font-weight:600;border-bottom:1px solid #cbd5e1}
img{max-width:100%;height:auto}
.warn{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:10px 14px;
font-size:13px;margin:12px 0}
"""


def write_html(path: str | Path, title: str, subtitle: str,
               curves: dict[str, pd.Series], metrics: dict[str, dict],
               notes: list[str] | None = None,
               extra_tables: dict[str, pd.DataFrame] | None = None) -> Path:
    """Write a single-file HTML report (chart embedded) and return its path."""
    parts = [
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(title)}</h1><p class='sub'>{html.escape(subtitle)}</p>",
    ]
    for note in notes or []:
        parts.append(f"<div class='warn'>{html.escape(note)}</div>")
    parts.append("<h2>資産推移とドローダウン</h2><div class='card'>"
                 f"<img alt='equity chart' src='data:image/png;base64,{_chart(curves)}'></div>")
    parts.append("<h2>成績</h2><div class='card'>"
                 + metrics_table(metrics).to_html(border=0, escape=True) + "</div>")
    for name, table in (extra_tables or {}).items():
        parts.append(f"<h2>{html.escape(name)}</h2><div class='card'>"
                     + table.to_html(border=0, index=False, escape=True) + "</div>")
    parts.append("<p class='sub'>過去データによる検証結果であり、将来の利益を保証しない。"
                 "コスト設定は仮定値。</p></main></body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Screener report
# --------------------------------------------------------------------------

_SCREEN_CSS = """
main{max-width:1280px}
.res th,.res td{padding:6px 7px}
.res th:nth-child(-n+3),.res td:nth-child(-n+3){text-align:left}
.res td:nth-child(2),.res td:nth-child(3){max-width:170px;overflow:hidden;text-overflow:ellipsis}
td.ok{color:#047857;font-weight:600}td.ng{color:#b91c1c}td.na{color:#94a3b8}
tr.cand td{background:#ecfdf5}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.pill.ok{background:#d1fae5;color:#065f46}.pill.ng{background:#fee2e2;color:#991b1b}
ul{margin:6px 0 0 18px;padding:0;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.stat{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px}.stat span{font-size:12px;color:#64748b}
"""
_TREND = {1: "上昇", 0: "なし", -1: "下降"}


def _cell(value, kind: str, passed=None) -> str:
    cls = "na" if passed is None else ("ok" if passed else "ng")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "<td class='na'>-</td>"
    text = {"pct": lambda v: f"{v * 100:+.0f}%", "num": lambda v: f"{v:.2f}",
            "int": lambda v: f"{int(v)}", "price": lambda v: f"{v:,.2f}",
            "str": str}[kind](value)
    return f"<td class='{cls}'>{html.escape(text)}</td>"


def write_screen_html(path: str | Path, result, top: int = 50) -> Path:
    """Write the screener result (``tradelab.screen.ScreenResult``) as HTML."""
    cfg, t = result.config, result.table
    fund = result.use_fundamentals
    n_cand = int(t["candidate"].sum())
    max_score = int(t["max_score"].iloc[0])
    shown = t[t["candidate"] | (t.index < top)]

    p = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         f"<title>銘柄スクリーニング</title><style>{_CSS}{_SCREEN_CSS}</style></head><body><main>",
         "<h1>銘柄スクリーニング: " + ("CAN SLIM + ダウ理論" if fund else "ダウ理論 + テクニカル条件") + "</h1>",
         f"<p class='sub'>対象: {html.escape(result.universe)} / 価格基準日: {result.asof.date()} / "
         f"評価できた銘柄: {len(t)} / 評価不能: {len(result.failures)}</p>"]

    p.append("<div class='grid'>")
    p.append(f"<div class='stat'><b>{n_cand}</b><span>全条件を満たす候補</span></div>")
    p.append(f"<div class='stat'><b>{int((t['score'] >= max_score - 1).sum())}</b>"
             f"<span>{max_score - 1}条件以上を満たす銘柄</span></div>")
    for label, m in result.market.items():
        pill = "<span class='pill ok'>合格</span>" if m["pass"] else "<span class='pill ng'>不合格</span>"
        p.append(f"<div class='stat'><b>{html.escape(label)} {pill}</b><span>ダウ理論トレンド: "
                 f"{_TREND[m['trend']]}（{m['bars_in_trend']}営業日継続） / 200日平均より"
                 f"{'上' if m['above_sma200'] else '下'}</span></div>")
    p.append("</div>")
    if not result.market_pass:
        p.append("<div class='warn'>市場全体の条件（M）が不合格。CAN SLIM では市場が上昇トレンドにない"
                 "局面での新規買いを避ける。下表は条件の合格数による順位であり、買い候補ではない。</div>")

    rules = [
        ("DOW", f"ダウ理論の上昇トレンド（直近の押し安値が切り上がり、終値が直近の戻り高値を上抜け。"
                f"終値が直近の押し安値を割るまで継続。スイング判定は前後{cfg.dow_swing}本）"),
        ("N", f"終値が52週高値から {cfg.n_max_below_high:.0%} 以内"),
        ("S", f"直近50日の上昇日出来高 ÷ 下落日出来高が {cfg.s_min_updown_volume} 以上"),
        ("L", f"相対的強さ（直近12か月の騰落率、直近3か月を2倍に加重）が対象内の上位 "
              f"{100 - cfg.l_min_rs_rank:.0f}% 以内"),
        ("M", "市場指数がダウ理論の上昇トレンドにあり、かつ200日移動平均より上（全銘柄共通の前提条件）"),
    ]
    if fund:
        rules = [
            ("C", f"直近四半期の希薄化後EPSが前年同期比 +{cfg.c_min_eps_growth:.0%} 以上"),
            ("A", f"年次EPSの年平均成長率が +{cfg.a_min_eps_cagr:.0%} 以上、毎年増益、"
                  f"ROE {cfg.a_min_roe:.0%} 以上"),
        ] + rules + [("I", "機関投資家の保有動向: データを取得できないため未評価")]
    p.append("<h2>判定ルール</h2><div class='card'><ul>"
             + "".join(f"<li><b>{k}</b>: {html.escape(v)}</li>" for k, v in rules) + "</ul></div>")

    head = ["銘柄", "名称", "業種", "合格数"]
    if fund:
        head += ["C: 四半期EPS", "四半期売上", "A: EPS年成長", "ROE"]
    head += ["N: 高値比", "S: 出来高比", "L: RS順位", "DOW: トレンド", "継続日数", "終値"]
    p.append(f"<h2>結果（候補 + 合格数上位 {top} 銘柄）</h2><div class='card'><table class='res'><thead><tr>"
             + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>")
    for _, r in shown.iterrows():
        cells = [f"<td><b>{html.escape(str(r['symbol']))}</b></td>",
                 f"<td>{html.escape(str(r.get('name') or ''))}</td>",
                 f"<td>{html.escape(str(r.get('sector') or ''))}</td>",
                 f"<td>{int(r['score'])}/{max_score}</td>"]
        if fund:
            cells += [_cell(r["eps_growth_q"], "pct", bool(r["C"])),
                      _cell(r["revenue_growth_q"], "pct"),
                      _cell(r["eps_cagr"], "pct", bool(r["A"])),
                      _cell(r["roe"], "pct", bool(r["A"]))]
        cells += [_cell(-r["below_high"], "pct", bool(r["N"])),
                  _cell(r["updown_volume"], "num", bool(r["S"])),
                  _cell(r["rs_rank"], "int", bool(r["L"])),
                  _cell(_TREND[int(r["dow_trend"])], "str", bool(r["DOW"])),
                  _cell(r["dow_bars"], "int"), _cell(r["close"], "price")]
        p.append(f"<tr class='{'cand' if r['candidate'] else ''}'>" + "".join(cells) + "</tr>")
    p.append("</tbody></table></div>")

    if len(result.failures):
        p.append("<h2>評価できなかった銘柄</h2><div class='card'>"
                 + result.failures.rename(columns={"symbol": "銘柄", "reason": "理由"})
                 .to_html(border=0, index=False, escape=True) + "</div>")
    p.append("<p class='sub'>条件に合致した銘柄の一覧であり、売買の推奨ではない。このルールの有効性は"
             "過去データで検証されていない。決算数値はデータ提供元の値で、誤りや遅延を含みうる。</p>"
             "</main></body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(p), encoding="utf-8")
    return path
