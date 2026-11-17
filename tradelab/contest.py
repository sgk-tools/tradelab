"""Friends' trading contest: orders arrive through a Google Form, results are
computed here from market data so nobody can choose a price after the fact.

Flow: Google Form -> response sheet -> a published "public" tab (CSV) that
holds only verified rows (the passcode check is done inside the sheet and
the passcode itself is never published) -> ``load_orders`` -> ``simulate``.

Rules (same as the page's demo trading):
- market: first session open after the order was submitted
- limit:  buy when the day's low reaches the price (open if better),
          sell when the high reaches it
- stop:   buy when the high reaches the price, sell when the low reaches it
- limit / stop orders lapse after ``ORDER_DAYS`` sessions
- optional stop-loss / take-profit on a buy sells the whole position on the
  day the low / high reaches it (stop first when both; open if gapped)
- Japanese stocks trade in 100-share lots, crypto in 0.0001 units, others 1
- a buy that the cash cannot cover at the fill price is rejected
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ORDER_DAYS = 60
JST = ZoneInfo("Asia/Tokyo")
NY = ZoneInfo("America/New_York")
SIDES = {"買い": "buy", "買": "buy", "buy": "buy", "売り": "sell", "売": "sell", "sell": "sell",
         "取消": "cancel", "取り消し": "cancel", "cancel": "cancel"}
TYPES = {"成行": "market", "": "market", "market": "market", "指値": "limit", "limit": "limit",
         "逆指値": "stop", "stop": "stop"}
FEES = {"crypto": 0.0015}
DEFAULT_FEE = 0.001


@dataclass
class Order:
    id: int
    player: str
    ts: datetime                 # submission time (UTC)
    symbol: str
    side: str                    # buy / sell / cancel
    type: str = "market"
    qty: float = 0.0
    px: float | None = None
    st: float | None = None      # stop-loss for a buy
    tg: float | None = None      # take-profit for a buy
    target: int | None = None    # order id to cancel
    key: str = ""                # list the symbol belongs to (us / jp / etf / crypto)
    status: str = "open"         # open / filled / cancelled / expired / rejected
    reason: str = ""
    fill_date: str | None = None
    fill_px: float | None = None


def load_config(path: str | Path) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}
    cfg.setdefault("capital", 10_000_000)
    cfg.setdefault("start", "2000-01-01")
    return cfg


def fetch_csv(url: str, timeout: float = 30.0) -> str:
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def _num(text) -> float | None:
    s = str(text or "").strip().replace(",", "").replace("，", "").replace("$", "").replace("¥", "").replace("円", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _time(text: str, tz: ZoneInfo) -> datetime | None:
    s = str(text or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=tz).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def load_orders(text: str, symbol_list: dict[str, str], tz: str = "Asia/Tokyo",
                start: str | None = None) -> tuple[list[Order], int]:
    """Parse the published CSV (header row + one row per verified submission).

    Columns by position: time, nickname (blank when the passcode did not
    match), symbol, side, type, quantity, price, stop-loss, take-profit,
    order number to cancel. Returns (orders, rows dropped as unverified).
    Order ids are the row numbers (1 = first data row).
    """
    zone = ZoneInfo(tz)
    rows = list(csv.reader(io.StringIO(text)))
    orders, unverified = [], 0
    begin = datetime.fromisoformat(start).replace(tzinfo=zone).astimezone(timezone.utc) if start else None
    for i, row in enumerate(rows[1:], 1):
        row = (row + [""] * 10)[:10]
        if not any(c.strip() for c in row):
            continue
        ts = _time(row[0], zone)
        name = row[1].strip()
        if not name:
            unverified += 1
            continue
        side = SIDES.get(row[3].strip().lower() if row[3].strip().isascii() else row[3].strip())
        o = Order(id=i, player=name[:20], ts=ts or datetime.now(timezone.utc),
                  symbol=row[2].strip().upper(), side=side or "",
                  type=TYPES.get(row[4].strip(), ""), qty=_num(row[5]) or 0.0, px=_num(row[6]),
                  st=_num(row[7]), tg=_num(row[8]),
                  target=int(_num(row[9])) if _num(row[9]) else None)
        if ts is None:
            o.status, o.reason = "rejected", "受付時刻を読めない"
        elif begin and ts < begin:
            o.status, o.reason = "rejected", "大会開始前の注文"
        elif not side:
            o.status, o.reason = "rejected", "売買の区分が不明"
        elif side != "cancel":
            o.key = symbol_list.get(o.symbol, "")
            if not o.key:
                o.status, o.reason = "rejected", "対象外の銘柄（4市場の一覧にない）"
            elif not o.type:
                o.status, o.reason = "rejected", "注文の種類が不明"
            elif o.type != "market" and not (o.px and o.px > 0):
                o.status, o.reason = "rejected", "指値・逆指値の価格がない"
            else:
                o.qty = round_lot(o.key, o.symbol, o.qty)
                if o.qty <= 0:
                    o.status, o.reason = "rejected", f"数量が売買単位（{lot_of(o.key, o.symbol):g}）未満"
        elif not o.target:
            o.status, o.reason = "rejected", "取消する注文番号がない"
        orders.append(o)
    return orders, unverified


def lot_of(key: str, symbol: str) -> float:
    if key == "crypto" or symbol.endswith("-USD"):
        return 0.0001
    if key == "jp":
        return 100
    return 1


def round_lot(key: str, symbol: str, qty: float) -> float:
    lot = lot_of(key, symbol)
    if lot < 1:
        return float(np.floor(qty * 10000) / 10000)
    return float(np.floor(qty / lot) * lot)


def min_fill_date(ts: datetime, key: str, symbol: str) -> str:
    """First session date whose open comes after ``ts``."""
    def local(z):
        return ts.astimezone(z)
    if key == "crypto" or symbol.endswith("-USD"):
        d = ts.astimezone(timezone.utc).date() + timedelta(days=1)       # bars open 00:00 UTC
    elif symbol.endswith(".T"):
        t = local(JST)
        d = t.date() + timedelta(days=1) if (t.hour, t.minute) >= (9, 0) else t.date()
    else:
        t = local(NY)
        d = t.date() + timedelta(days=1) if (t.hour, t.minute) >= (9, 30) else t.date()
    return d.isoformat()


def currency_of(symbol: str) -> str:
    return "JPY" if symbol.endswith(".T") else "USD"


@dataclass
class Player:
    name: str
    cash: float
    pos: dict = field(default_factory=dict)      # symbol -> dict(q, cost, usd, key, st, tg, first)
    equity: dict = field(default_factory=dict)   # date -> yen
    trades: list = field(default_factory=list)
    realized: float = 0.0


def simulate(orders: list[Order], bars: Callable[[str], pd.DataFrame], usdjpy: pd.Series,
             capital: float, until: str | None = None, exits: dict | None = None,
             names: list[str] | None = None) -> dict[str, Player]:
    """Replay every player's orders against daily bars; returns players by name.

    ``exits`` maps (player, symbol) to a stop/target setting edited on a
    holding: {"on", "st", "tg", "ts"}; it replaces the last buy's own setting
    from the first session after it was saved. ``names`` adds players that
    have not ordered yet.
    """
    exits = exits or {}
    players: dict[str, Player] = {n: Player(n, capital) for n in (names or [])}
    for o in orders:
        if o.player not in players:
            players[o.player] = Player(o.player, capital)
    live = [o for o in orders if o.status == "open" and o.side in ("buy", "sell")]
    cancels = [o for o in orders if o.status == "open" and o.side == "cancel"]
    by_id = {o.id: o for o in orders}
    for c in cancels:
        t = by_id.get(c.target)
        if not t or t.player != c.player or t.side == "cancel":
            c.status, c.reason = "rejected", "取消できる自分の注文がない"
        else:
            c.status = "filled"                       # recorded; applied below by date
            t.cancel_from = min_fill_date(c.ts, t.key or "", t.symbol)
    data: dict[str, pd.DataFrame] = {}
    for o in live:
        if o.symbol not in data:
            try:
                data[o.symbol] = bars(o.symbol)
            except Exception:
                data[o.symbol] = None
    fx = usdjpy.dropna().sort_index()

    def rate(cur: str, day: str) -> float:
        if cur == "JPY":
            return 1.0
        s = fx[fx.index <= pd.Timestamp(day)]
        return float(s.iloc[-1]) if len(s) else float(fx.iloc[0])

    days = sorted({d.strftime("%Y-%m-%d") for df in data.values() if df is not None for d in df.index})
    if until:
        days = [d for d in days if d <= until]
    first_day = {}
    for o in live:
        o.min_date = min_fill_date(o.ts, o.key, o.symbol)
        o.sessions = 0
        first_day[o.player] = min(first_day.get(o.player, o.min_date), o.min_date)
    rows = {s: (df, {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)})
            for s, df in data.items() if df is not None}
    last_close: dict[str, float] = {}

    for day in days:
        todays = [o for o in live if o.status == "open" and o.min_date <= day]
        todays.sort(key=lambda o: (o.ts, o.id))
        for o in todays:
            if o.symbol not in rows or day not in rows[o.symbol][1]:
                continue
            if getattr(o, "cancel_from", None) and day >= o.cancel_from:
                o.status = "cancelled"
                continue
            df, idx = rows[o.symbol]
            b = df.iloc[idx[day]]
            o.sessions += 1
            if o.sessions > ORDER_DAYS:
                o.status, o.reason = "expired", f"{ORDER_DAYS}取引日で失効"
                continue
            op, hi, lo = float(b["open"]), float(b["high"]), float(b["low"])
            buy, px = o.side == "buy", None
            if o.type == "market":
                px = op
            elif o.type == "limit":
                if buy and lo <= o.px:
                    px = min(op, o.px)
                elif not buy and hi >= o.px:
                    px = max(op, o.px)
            elif o.type == "stop":
                if buy and hi >= o.px:
                    px = max(op, o.px)
                elif not buy and lo <= o.px:
                    px = min(op, o.px)
            if px is None:
                continue
            _execute(players[o.player], o, day, px, rate(currency_of(o.symbol), day))
        for p in players.values():
            for sym, P in list(p.pos.items()):
                if P["q"] <= 1e-12 or sym not in rows or day not in rows[sym][1] or day < P["first"]:
                    continue
                df, idx = rows[sym]
                b = df.iloc[idx[day]]
                op, hi, lo = float(b["open"]), float(b["high"]), float(b["low"])
                entry_day = day == P["first"]
                st, tg = P.get("st"), P.get("tg")
                e = exits.get((p.name, sym))
                if e and e["ts"] > P.get("ts", e["ts"]) and day >= min_fill_date(e["ts"], P["key"], sym):
                    st, tg = (e.get("st"), e.get("tg")) if e.get("on") else (None, None)
                px = why = None
                if st is not None and not entry_day and op <= st:
                    px, why = op, "stop"
                elif st is not None and lo <= st:
                    px, why = st, "stop"
                elif tg is not None and not entry_day and op >= tg:
                    px, why = op, "target"
                elif tg is not None and hi >= tg:
                    px, why = tg, "target"
                if why:
                    auto = Order(id=0, player=p.name, ts=datetime.now(timezone.utc), symbol=sym,
                                 side="sell", qty=P["q"], key=P["key"])
                    _execute(p, auto, day, px, rate(currency_of(sym), day), why)
        for sym, (df, idx) in rows.items():
            if day in idx:
                last_close[sym] = float(df["close"].iloc[idx[day]])
        for p in players.values():
            if day < first_day.get(p.name, "9999"):
                continue
            value = sum(P["q"] * last_close.get(s, P["cost"] / max(P["q"], 1e-12) / rate(currency_of(s), day))
                        * rate(currency_of(s), day) for s, P in p.pos.items() if P["q"] > 1e-12)
            p.equity[day] = p.cash + value
    for p in players.values():
        p.last_close = last_close
        p.rate_now = {"USD": rate("USD", days[-1]) if days else float(fx.iloc[-1]), "JPY": 1.0}
    return players


def _execute(p: Player, o: Order, day: str, px: float, rate: float, why: str = "manual") -> None:
    fee = FEES.get(o.key, DEFAULT_FEE)
    P = p.pos.get(o.symbol)
    if o.side == "buy":
        cost = o.qty * px * rate * (1 + fee)
        if cost > p.cash + 1e-6:
            o.status, o.reason = "rejected", "約定時の資金不足"
            return
        if P is None or P["q"] <= 1e-12:
            P = p.pos[o.symbol] = {"q": 0.0, "cost": 0.0, "usd": 0.0, "key": o.key, "first": day,
                                    "st": None, "tg": None}
        P["q"] += o.qty
        P["cost"] += cost
        P["usd"] += o.qty * px
        P["st"], P["tg"], P["ts"] = o.st, o.tg, o.ts
        p.cash -= cost
    else:
        q = min(o.qty, P["q"]) if P else 0.0
        if q <= 1e-12:
            o.status, o.reason = "rejected", "売る株がない"
            return
        avg = P["cost"] / P["q"]
        proceeds = q * px * rate * (1 - fee)
        pnl = proceeds - avg * q
        P["usd"] -= P["usd"] / P["q"] * q
        P["cost"] -= avg * q
        P["q"] = round(P["q"] - q, 8)
        p.cash += proceeds
        p.realized += pnl
        p.trades.append({"d": day, "s": o.symbol, "q": q, "p": round(px, 6), "pnl": round(pnl),
                         "pct": round(pnl / (avg * q), 4), "why": why})
    o.status, o.fill_date, o.fill_px = "filled", day, round(px, 6)


def period_starts(asof: str) -> dict[str, str]:
    d = date.fromisoformat(asof)
    return {"week": (d - timedelta(days=d.weekday())).isoformat(),
            "month": d.replace(day=1).isoformat(), "year": d.replace(month=1, day=1).isoformat()}


def leaderboard(players: dict[str, Player], capital: float, asof: str) -> dict:
    """Return rates over week (Mon), month, year and all time, per player."""
    starts = period_starts(asof)
    out = []
    for p in players.values():
        eq = dict(sorted(p.equity.items()))
        dates = [d for d in eq if d <= asof]
        now = eq[dates[-1]] if dates else capital

        def base(start):
            before = [d for d in dates if d < start]
            return eq[before[-1]] if before else capital
        rets = {k: now / base(v) - 1 for k, v in starts.items()}
        rets["all"] = now / capital - 1
        hold = []
        for s, P in p.pos.items():
            if P["q"] <= 1e-12:
                continue
            cur = currency_of(s)
            close = p.last_close.get(s)
            val = P["q"] * (close or 0) * p.rate_now[cur]
            hold.append({"s": s, "q": P["q"], "avg": round(P["usd"] / P["q"], 6), "cl": close, "cu": cur,
                         "val": round(val), "pnl": round(val - P["cost"]), "st": P.get("st"), "tg": P.get("tg")})
        series = [[d, round(eq[d])] for d in dates][-400:]
        out.append({"name": p.name, "equity": round(now), "cash": round(p.cash),
                    "realized": round(p.realized), "ret": {k: round(v, 5) for k, v in rets.items()},
                    "hold": sorted(hold, key=lambda h: -h["val"]), "trades": p.trades[-30:][::-1],
                    "series": series})
    return {"asof": asof, "capital": capital, "starts": starts, "players": out}


# ---- Supabase source (accounts with ID and password) -------------------------------

def fetch_supabase(url: str, key: str, table: str, select: str = "*", order: str | None = None,
                   page: int = 1000, timeout: float = 30.0) -> list[dict]:
    """All rows of ``table`` through the REST API with the service key (bypasses row rules)."""
    import requests
    rows, start = [], 0
    while True:
        params = {"select": select}
        if order:
            params["order"] = order
        r = requests.get(f"{url.rstrip('/')}/rest/v1/{table}", params=params, timeout=timeout,
                         headers=_headers(key, start, page))
        r.raise_for_status()
        chunk = r.json()
        rows += chunk
        if len(chunk) < page:
            return rows
        start += page


def _headers(key: str, start: int, page: int) -> dict:
    h = {"apikey": key, "Range-Unit": "items", "Range": f"{start}-{start + page - 1}"}
    if not key.startswith("sb_"):          # legacy JWT keys also go in Authorization
        h["Authorization"] = f"Bearer {key}"
    return h


def _utc(text: str | None) -> datetime | None:
    if not text:
        return None
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def orders_from_rows(rows: list[dict], names: dict[str, str], symbol_list: dict[str, str],
                     start: str | None = None) -> list[Order]:
    """Orders table rows (id, user_id, created_at, symbol, side, type, qty, px, st, tg, target)."""
    begin = datetime.fromisoformat(start).replace(tzinfo=JST).astimezone(timezone.utc) if start else None
    out = []
    for r in sorted(rows, key=lambda r: r["id"]):
        name = names.get(r.get("user_id"))
        if not name:
            continue
        o = Order(id=int(r["id"]), player=name, ts=_utc(r.get("created_at")) or datetime.now(timezone.utc),
                  symbol=str(r.get("symbol") or "").upper(), side=r.get("side") or "",
                  type=r.get("type") or "market", qty=float(r.get("qty") or 0),
                  px=r.get("px"), st=r.get("st"), tg=r.get("tg"),
                  target=int(r["target"]) if r.get("target") else None)
        if begin and o.ts < begin:
            o.status, o.reason = "rejected", "大会開始前の注文"
        elif o.side == "cancel":
            if not o.target:
                o.status, o.reason = "rejected", "取消する注文番号がない"
        elif o.side not in ("buy", "sell"):
            o.status, o.reason = "rejected", "売買の区分が不明"
        else:
            o.key = symbol_list.get(o.symbol, "")
            if not o.key:
                o.status, o.reason = "rejected", "対象外の銘柄（4市場の一覧にない）"
            elif o.type not in ("market", "limit", "stop"):
                o.status, o.reason = "rejected", "注文の種類が不明"
            elif o.type != "market" and not (o.px and o.px > 0):
                o.status, o.reason = "rejected", "指値・逆指値の価格がない"
            else:
                o.qty = round_lot(o.key, o.symbol, o.qty)
                if o.qty <= 0:
                    o.status, o.reason = "rejected", f"数量が売買単位（{lot_of(o.key, o.symbol):g}）未満"
        out.append(o)
    return out


def build_supabase(cfg: dict, key: str, symbol_list: dict[str, str],
                   bars: Callable[[str], pd.DataFrame], usdjpy: pd.Series,
                   fetch: Callable | None = None) -> dict:
    fetch = fetch or (lambda table, **kw: fetch_supabase(cfg["supabase_url"], key, table, **kw))
    profiles = fetch("profiles", select="id,name")
    names = {p["id"]: p["name"] for p in profiles}
    orders = orders_from_rows(fetch("orders", order="id.asc"), names, symbol_list, cfg.get("start"))
    exits = {}
    for e in fetch("exits"):
        if e.get("user_id") in names:
            exits[(names[e["user_id"]], str(e["symbol"]).upper())] = {
                "on": bool(e.get("enabled")), "st": e.get("st"), "tg": e.get("tg"), "ts": _utc(e.get("set_at"))}
    players = simulate(orders, bars, usdjpy, cfg["capital"], exits=exits, names=list(names.values()))
    days = sorted({d for p in players.values() for d in p.equity})
    asof = days[-1] if days else datetime.now(JST).date().isoformat()
    board = leaderboard(players, cfg["capital"], asof)
    board["orders"] = _order_rows(orders)
    board.update(unverified=0, form="", start=cfg.get("start"), jpLot=100)
    return board


def favorite_symbols(cfg: dict, key: str, fetch: Callable | None = None) -> list[str]:
    fetch = fetch or (lambda table, **kw: fetch_supabase(cfg["supabase_url"], key, table, **kw))
    return sorted({str(r["symbol"]).upper() for r in fetch("favorites", select="symbol")})


def _order_rows(orders: list[Order]) -> list[dict]:
    return [{"id": o.id, "p": o.player, "t": o.ts.astimezone(JST).strftime("%m-%d %H:%M"),
             "s": o.symbol, "side": o.side, "type": o.type, "q": o.qty, "px": o.px,
             "st": o.status, "why": o.reason, "fd": o.fill_date, "fp": o.fill_px}
            for o in orders][-300:][::-1]


def build(cfg: dict, csv_text: str, symbol_list: dict[str, str],
          bars: Callable[[str], pd.DataFrame], usdjpy: pd.Series) -> dict:
    orders, unverified = load_orders(csv_text, symbol_list, cfg.get("tz", "Asia/Tokyo"), cfg.get("start"))
    players = simulate(orders, bars, usdjpy, cfg["capital"])
    days = sorted({d for p in players.values() for d in p.equity})
    asof = days[-1] if days else datetime.now(JST).date().isoformat()
    board = leaderboard(players, cfg["capital"], asof)
    board["orders"] = _order_rows(orders)
    board.update(unverified=unverified, form=cfg.get("form_url", ""), start=cfg.get("start"),
                 jpLot=100)
    return board
