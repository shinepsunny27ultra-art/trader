"""Gold book: at most 3 gold-linked NSE stocks a day, traded intraday on the gold price.

Research: research/gold_book_2026_11/ (gold_lab.py, combine.py; docs/research-log.md
2026-11-23).  Two rules held up in DEV (2016-07..2023-12) and HOLD (2024-01..2026-11):

  LENDERS  gold moved >= 0.5% overnight (15:30 IST yesterday -> 09:00 IST today):
           trade the gold lenders (MUTHOOTFIN, MANAPPURAM, IIFL) in gold's direction.
           Their intraday move follows the overnight gold move (DEV corr +0.12;
           jewellers ~0).  DEV +16.5 bps/trade (t 2.3), HOLD +18.6 (t 2.4).
           Needs only the 09:00 gold price: entered in the pre-open (09:01-09:07).
  GAP FADE fade the part of a gold stock's opening gap that gold's overnight move
           does not explain: resid = gap0 - beta * g_on, |resid| >= 1%, largest first,
           fills the slots the lenders leave.  DEV +26.0 (t 2.7), HOLD +25.2 (t 2.3);
           the raw gap without the gold adjustment: HOLD +13.0.  Needs today's open:
           entered ~09:16-09:20 (the backtest entered at the open; untested delay).

Combined, at the backtest's open entries: +16.5 bps/trade, Sharpe ~1.1, max drawdown
~47% of the book's capital over 2016-26.  A weak edge next to the gap book (Sharpe ~5):
it is a separate, paper-tracked section.  Trades as the main book: stop 0.75 x 14-day
ATR, exit 15:15, Rs 1.25 lakh a name.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gold_book"
PICKS = OUT / "picks.json"
PAPER = OUT / "paper_book.csv"
GOLD_MARKS = OUT / "gold_marks.csv"
IST = "Asia/Kolkata"

UNIVERSE = ["TITAN", "KALYANKJIL", "SENCO", "THANGAMAYL", "RAJESHEXPO", "PCJEWELLER", "SKYGOLD",
            "SHANTIGOLD", "GOLDIAM", "TBZ", "VAIBHAVGBL", "BLUESTONE", "MOTISONS", "PNGJL",
            "MUTHOOTFIN", "MANAPPURAM", "IIFL"]
LENDERS = ("MUTHOOTFIN", "MANAPPURAM", "IIFL")
K = 3                       # at most three names a day
GOLD_MOVE = 0.005           # overnight gold move that triggers the lender trades
RESID_MIN = 0.01            # unexplained opening gap that triggers a fade
STOP_ATR = 0.75
CAPITAL_PER_NAME = 125_000.0
MIN_VALUE = 1e8             # Rs 10 crore 20-session mean traded value
MIN_PRICE = 50.0
BETA_WINDOW, BETA_MIN = 250, 60
GAP_FADE_ENTRY_BAR = 1      # 09:20 bar open: the list needs the 09:15 open first

# Backtest summary shown on the dashboard (research/gold_book_2026_11/combine.json).
EVIDENCE = {
    "dev": {"period": "2016-07 .. 2023-12", "net_bps_per_trade": 15.9, "t": 2.51, "trades": 2938},
    "hold": {"period": "2024-01 .. 2026-11", "net_bps_per_trade": 17.7, "t": 2.39, "trades": 1636},
    "lenders": {"dev": 16.5, "hold": 18.6}, "gap_fade": {"dev": 26.0, "hold": 25.2},
    "sharpe": 1.06, "max_drawdown_pct": 47.1,
}


# ── gold prices at the NSE clock ────────────────────────────────────────────

def marks(prices: pd.Series, sessions, hhmm: str, max_age_hours: float = 6.0) -> pd.Series:
    """Gold at `hhmm` IST on each session: the last price at or before it (NaN if stale)."""
    p = prices.dropna().sort_index()
    h, m = map(int, hhmm.split(":"))
    stamps = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(IST) + pd.Timedelta(hours=h, minutes=m)
                               for d in sessions])
    if p.empty:
        return pd.Series(np.nan, index=pd.Index(sessions))
    idx = p.index.tz_convert(IST)
    pos = idx.searchsorted(stamps, side="right") - 1
    safe = np.clip(pos, 0, None)
    age = (stamps - idx[safe]).total_seconds() / 3600
    vals = np.where((pos >= 0) & (age < max_age_hours), p.to_numpy()[safe], np.nan)
    return pd.Series(vals, index=pd.Index(sessions))


def gold_marks(prices: pd.Series, sessions) -> pd.DataFrame:
    return pd.DataFrame({"p0900": marks(prices, sessions, "09:00"),
                         "p1530": marks(prices, sessions, "15:30")})


DUKA_URL = ("https://freeserv.dukascopy.com/2.0/index.php?path=chart%2Fjson3&instrument=XAU%2FUSD"
            "&offer_side=B&interval=1MIN&splits=true&stocks=true&limit=30000&time_direction=N"
            "&timestamp={ts}&jsonp=_cb")
DUKA_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://freeserv.dukascopy.com/2.0/?path=chart/index"}


def fetch_minutes(since: pd.Timestamp, *, pages: int = 6) -> pd.Series:
    """Spot gold (XAU/USD bid) 1-minute closes from `since` to now, UTC-stamped (bar start).

    The research's source (commodity/research/fetch_duka_chart.py).  Not Yahoo's GC=F:
    the continuous future jumps when it rolls (2026-10-26: -0.61% "overnight" against
    spot's -0.23%), which would fake lender trades.
    """
    import urllib.request

    ts, rows = int(pd.Timestamp(since).timestamp() * 1000), []
    for _ in range(pages):
        req = urllib.request.Request(DUKA_URL.format(ts=ts), headers=DUKA_HEADERS)
        body = urllib.request.urlopen(req, timeout=60).read().decode()
        page = [r for r in (json.loads(body[body.index("(") + 1:body.rindex(")")]) or []) if r]
        if not page:
            break
        rows += page
        nxt = int(page[-1][0]) + 60_000
        if nxt <= ts or len(page) < 30_000:
            break
        ts = nxt
    if not rows:
        raise RuntimeError("no spot gold minutes from Dukascopy")
    df = pd.DataFrame(rows).iloc[:, :5]
    s = pd.Series(df[4].astype(float).to_numpy(), index=pd.to_datetime(df[0], unit="ms", utc=True))
    return s[~s.index.duplicated(keep="last")].sort_index()


def marks_table(sessions, *, fetch: bool = True, log=print) -> pd.DataFrame:
    """Gold at 09:00 / 15:30 IST for `sessions`, from the stored table (data/gold_book/
    gold_marks.csv, seeded from the research's minute history) topped up from Dukascopy:
    the last few stored days are always re-read so a part-day mark is replaced."""
    sessions = pd.DatetimeIndex(pd.to_datetime(list(sessions)))
    try:
        table = pd.read_csv(GOLD_MARKS, index_col="session", parse_dates=["session"])
    except (OSError, ValueError):
        table = pd.DataFrame(columns=["p0900", "p1530"], dtype=float)
    if fetch:
        last = table.dropna().index.max() if not table.dropna().empty else sessions.max() - pd.Timedelta(days=20)
        recent = sessions[sessions >= min(last, sessions.max()) - pd.Timedelta(days=3)]
        if len(recent):
            minutes = fetch_minutes(recent.min().tz_localize(IST) - pd.Timedelta(days=1))
            fresh = gold_marks(minutes, recent)
            fresh.index = pd.DatetimeIndex(fresh.index)
            table = fresh.combine_first(table) if not table.empty else fresh
            table = table[~table.index.duplicated(keep="first")].sort_index()
            OUT.mkdir(parents=True, exist_ok=True)
            table.rename_axis("session").round(3).to_csv(GOLD_MARKS)
    return table.reindex(sessions)


# ── features: shared by the research lab and the live list ─────────────────

def frames(bhav: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (session x symbol) daily frames of the gold universe from NSE bhavcopies."""
    b = bhav[bhav["symbol"].isin(UNIVERSE)].drop_duplicates(["session", "symbol"], keep="last").copy()
    b["session"] = pd.to_datetime(b["session"])
    W = lambda c: b.pivot(index="session", columns="symbol", values=c)
    o = W("open")
    out = {k: W(k).reindex_like(o) for k in ("high", "low", "close", "prev_close", "value")}
    out["open"] = o
    out["be"] = (W("series") == "BE").astype(float).reindex_like(o).where(o.notna())
    return out


def features(fr: dict[str, pd.DataFrame], gold: pd.DataFrame) -> dict:
    """Everything the rules need, each row using only what is known at that session's entry."""
    o, h, l, c, pc, val, be = (fr[k] for k in ("open", "high", "low", "close", "prev_close", "value", "be"))
    ratio = o / pc
    bad = (ratio < 0.70) | (ratio > 1.35)                     # unadjusted prev_close on split days
    gap0 = (o / pc - 1).mask(bad)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()]).groupby(level=0).max() / pc
    atrp = tr.rolling(14, min_periods=10).mean().shift(1)
    stop = STOP_ATR * atrp
    long_ = ((c - o) / o * 1e4).where(~(l <= o * (1 - stop)), -stop * 1e4).mask(bad)
    short = ((o - c) / o * 1e4).where(~(h >= o * (1 + stop)), -stop * 1e4).mask(bad)
    tradable = ((val.rolling(20, min_periods=10).mean().shift(1) >= MIN_VALUE)
                & (c.shift(1) >= MIN_PRICE)
                & (be.rolling(20, min_periods=1).max().shift(1).fillna(0) == 0))
    g = gold.reindex(o.index)
    g_on = g["p0900"] / g["p1530"].shift(1) - 1
    g_tr20 = g["p0900"] / g["p1530"].shift(20) - 1
    g_ma50 = g["p0900"] / g["p1530"].rolling(50, min_periods=40).mean().shift(1) - 1
    x = g_on
    roll = dict(window=BETA_WINDOW, min_periods=BETA_MIN)
    cov = gap0.mul(x, axis=0).rolling(**roll).mean() - \
        gap0.rolling(**roll).mean().mul(x.rolling(**roll).mean(), axis=0)
    beta = cov.div(x.rolling(**roll).var(ddof=0), axis=0).shift(1)
    return dict(gap0=gap0, gap1=gap0.shift(1), long=long_, short=short, ok=tradable, has_open=o.notna(),
                beta=beta, atrp=atrp, g_on=g_on, g_tr20=g_tr20, g_ma50=g_ma50,
                close=c, value20=val.rolling(20, min_periods=10).mean().shift(1))


def select(f: dict, *, stage: str = "open") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(lender picks, gap-fade picks, side) for every session; side +1 long, -1 short.

    stage="preopen": the lender trades only (no opening prices yet).
    """
    ok, beta, on = f["ok"], f["beta"], f["g_on"]
    lender = pd.DataFrame({s: s in LENDERS for s in beta.columns}, index=beta.index)
    sig = pd.Series(np.where(on > GOLD_MOVE, 1, np.where(on < -GOLD_MOVE, -1, 0)), index=on.index)
    p_l = (lender & ok).mul(sig != 0, axis=0).astype(bool)
    if stage != "preopen":
        p_l &= f["has_open"]                                  # a name that never opened did not trade
    side_l = pd.DataFrame(np.repeat(sig.to_numpy()[:, None], beta.shape[1], 1),
                          index=beta.index, columns=beta.columns)
    resid = f["gap0"] - beta.mul(on, axis=0)
    if stage == "preopen":
        p_g = pd.DataFrame(False, index=beta.index, columns=beta.columns)
    else:
        cand = ok & f["has_open"] & beta.notna() & (resid.abs() >= RESID_MIN) & ~p_l
        left = (K - p_l.sum(axis=1)).clip(lower=0)
        rank = resid.abs().where(cand).rank(axis=1, ascending=False, method="first")
        p_g = rank.le(left, axis=0)
    side = side_l.where(p_l, -np.sign(resid))
    return p_l, p_g, side


def trend(f: dict, day) -> dict:
    """The gold-trend context for the dashboard."""
    t20, ma = f["g_tr20"].get(day, np.nan), f["g_ma50"].get(day, np.nan)
    label = ("UP" if t20 > 0 and ma > 0 else "DOWN" if t20 < 0 and ma < 0 else "MIXED") \
        if np.isfinite(t20) and np.isfinite(ma) else "unknown"
    return {"overnight_pct": _pct(f["g_on"].get(day)), "trend20_pct": _pct(t20),
            "vs_ma50_pct": _pct(ma), "trend": label}


def _pct(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, 2)


# ── the live list ───────────────────────────────────────────────────────────

@dataclass
class GoldPick:
    symbol: str
    side: str                  # LONG | SHORT
    rule: str                  # LENDERS | GAP_FADE
    reason: str
    prev_close: float
    open: float | None
    atr_pct: float
    stop_pct: float
    quantity: int
    entry: str                 # when to enter
    beta: float | None = None
    resid_pct: float | None = None


def history(session: date) -> pd.DataFrame:
    from nse_intraday_ai import nse_bhav

    return nse_bhav.load(session - timedelta(days=430), session - timedelta(days=1), series=("EQ", "BE"))


def build(session: date, *, stage: str, opens: dict[str, float] | None = None,
          bhav: pd.DataFrame | None = None, gold: pd.DataFrame | None = None, fetch: bool = True,
          keep: list[dict] | None = None) -> dict:
    """Today's gold list.  `opens` (symbol -> today's open) is needed for stage="open";
    `gold` is a marks table (p0900/p1530 by session), else `marks_table()`.
    `keep`: the lender picks already published at 09:01 — the 09:16 list keeps them as
    they were (people entered them) and fills only the slots left with gap fades."""
    bhav = history(session) if bhav is None else bhav
    fr = frames(bhav)
    day = pd.Timestamp(session)
    if day in fr["open"].index:
        raise ValueError(f"history already contains {session}")
    for k in fr:                                              # today's row: the open only
        fr[k] = pd.concat([fr[k], pd.DataFrame(np.nan, index=[day], columns=fr[k].columns)])
    for s, px in (opens or {}).items():
        s = s.removesuffix(".NS")
        if s in fr["open"].columns and px and math.isfinite(px):
            fr["open"].at[day, s] = float(px)
    fr["prev_close"].loc[day] = fr["close"].shift(1).loc[day]
    gm = (marks_table(fr["open"].index, fetch=fetch) if gold is None else gold).reindex(fr["open"].index)
    if not np.isfinite(gm["p0900"].iloc[-1]) or not np.isfinite(gm["p1530"].iloc[-2]):
        raise RuntimeError("no gold price at 09:00 today or 15:30 yesterday")
    f = features(fr, gm)
    p_l, p_g, side = select(f, stage=stage)
    if keep is not None:                                      # the 09:01 lender trades stand
        kept = {p["symbol"] for p in keep}
        p_l = p_l & False
        resid = f["gap0"] - f["beta"].mul(f["g_on"], axis=0)
        cand = f["ok"] & f["has_open"] & f["beta"].notna() & (resid.abs() >= RESID_MIN)
        cand.loc[day, [c for c in cand.columns if c in kept]] = False
        rank = resid.abs().where(cand).rank(axis=1, ascending=False, method="first")
        p_g = rank <= max(K - len(keep), 0)
    picks = [GoldPick(**{k: v for k, v in p.items() if k in GoldPick.__dataclass_fields__})
             for p in (keep or [])]
    for rule, mask in (("LENDERS", p_l), ("GAP_FADE", p_g)):
        names = [s for s in mask.columns if bool(mask.at[day, s])]
        if rule == "GAP_FADE":
            resid = f["gap0"] - f["beta"].mul(f["g_on"], axis=0)
            names.sort(key=lambda s: -abs(resid.at[day, s]))
        for s in names:
            sd = "LONG" if side.at[day, s] > 0 else "SHORT"
            pc = float(f["close"].shift(1).at[day, s])
            atrp = float(f["atrp"].at[day, s])
            op = fr["open"].at[day, s]
            op = None if not np.isfinite(op) else float(op)
            px = op or pc
            g_on = f["g_on"].at[day]
            if rule == "LENDERS":
                reason = f"gold {g_on * 100:+.2f}% overnight: lenders follow gold"
                entry = "pre-open 09:01-09:07 (MIS market order; fills at the open)"
                beta = resid_pct = None
            else:
                r = float(f["gap0"].at[day, s] - f["beta"].at[day, s] * g_on)
                reason = (f"opened {f['gap0'].at[day, s] * 100:+.2f}%, gold explains "
                          f"{f['beta'].at[day, s] * g_on * 100:+.2f}%: fade the {r * 100:+.2f}% rest")
                entry = "09:16-09:20 (MIS market order)"
                beta, resid_pct = round(float(f["beta"].at[day, s]), 3), round(r * 100, 2)
            picks.append(GoldPick(symbol=s, side=sd, rule=rule, reason=reason, prev_close=round(pc, 2),
                                  open=op, atr_pct=round(atrp * 100, 2), stop_pct=round(STOP_ATR * atrp * 100, 2),
                                  quantity=int(CAPITAL_PER_NAME // px), entry=entry, beta=beta,
                                  resid_pct=resid_pct))
    universe = []
    for s in fr["open"].columns:
        universe.append({"symbol": s, "lender": s in LENDERS, "tradable": bool(f["ok"].at[day, s]),
                         "prev_close": _round(f["close"].shift(1).at[day, s]),
                         "gap_pct": _pct(f["gap0"].at[day, s]),
                         "beta": _round(f["beta"].at[day, s], 3),
                         "turnover_cr": _round(f["value20"].at[day, s] / 1e7, 1)})
    return {"session": session.isoformat(), "stage": stage,
            "generated_at": pd.Timestamp.now(tz=IST).isoformat(timespec="seconds"),
            "gold": trend(f, day), "picks": [asdict(p) for p in picks[:K]], "universe": universe,
            "rules": {"gold_move_pct": GOLD_MOVE * 100, "resid_min_pct": RESID_MIN * 100, "max_names": K,
                      "stop_atr": STOP_ATR, "capital_per_name": CAPITAL_PER_NAME}}


def _round(x, n: int = 2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def dated(session: date) -> Path:
    return OUT / "history" / f"picks_{session:%Y%m%d}.json"


def save(payload: dict) -> None:
    from nse_intraday_ai.atomic_io import atomic_write_json

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PICKS, payload)
    atomic_write_json(dated(date.fromisoformat(payload["session"])), payload)


def load(session: date | None = None) -> dict | None:
    """Today's list (picks.json), or a given session's from the dated history."""
    path = PICKS if session is None else dated(session)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if session is None or payload.get("session") == session.isoformat() else None


def remote_payload(session: date) -> dict | None:
    """A session's list as published on GitHub `main` (the other machine may have made it)."""
    from nse_intraday_ai import cloud_sync

    for path in (dated(session), PICKS):
        raw = cloud_sync.read_remote(path)
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        if payload and payload.get("session") == session.isoformat():
            return payload
    return None


# ── the paper record ────────────────────────────────────────────────────────

def simulate(pick: dict, bars: pd.DataFrame, entry_bar: int, square_off_bar: int,
             open_price: float | None = None) -> dict | None:
    """One trade on 5-minute bars (0 = 09:15): enter at `entry_bar`'s open (the official
    open for bar 0 when known), stop STOP_ATR x ATR away, exit at the 15:15 bar's open."""
    from nse_intraday_ai.costs import round_trip_cost

    if bars.empty or entry_bar not in bars.index:
        return None
    entry = float(open_price) if (entry_bar == 0 and open_price) else float(bars.at[entry_bar, "open"])
    if not math.isfinite(entry) or entry <= 0:
        return None
    qty = int(CAPITAL_PER_NAME // entry)
    long_ = pick["side"] == "LONG"
    dist = entry * pick["stop_pct"] / 100
    stop = entry - dist if long_ else entry + dist
    exit_price, reason, exit_bar = None, "SQUARE_OFF", square_off_bar
    for bar in range(entry_bar, square_off_bar):
        if bar not in bars.index:
            continue
        hit = bars.at[bar, "low"] <= stop if long_ else bars.at[bar, "high"] >= stop
        if hit:
            opened = float(bars.at[bar, "open"])
            exit_price = stop if bar == entry_bar else (min(stop, opened) if long_ else max(stop, opened))
            reason, exit_bar = "STOP", bar
            break
    if exit_price is None:
        if square_off_bar in bars.index and math.isfinite(bars.at[square_off_bar, "open"]):
            exit_price = float(bars.at[square_off_bar, "open"])
        else:
            closes = bars.loc[bars.index < square_off_bar, "close"].dropna()
            if closes.empty:
                return None
            exit_price, reason, exit_bar = float(closes.iloc[-1]), "LAST_BAR", int(closes.index[-1])
    gross = (exit_price - entry) * qty * (1 if long_ else -1)
    costs = round_trip_cost(entry, exit_price, qty, slippage_bps_per_leg=3.0).total
    minutes = 555 + 5 * exit_bar
    return {"symbol": pick["symbol"], "side": pick["side"], "rule": pick["rule"],
            "entry": round(entry, 2), "exit": round(exit_price, 2), "quantity": qty, "exit_reason": reason,
            "exit_time": f"{minutes // 60:02d}:{minutes % 60:02d}", "gross": round(gross, 2),
            "costs": round(costs, 2), "net": round(gross - costs, 2)}


def entry_bar(rule: str) -> int:
    return 0 if rule == "LENDERS" else GAP_FADE_ENTRY_BAR


def now_ist() -> datetime:
    return datetime.now(pd.Timestamp.now(tz=IST).tzinfo)
