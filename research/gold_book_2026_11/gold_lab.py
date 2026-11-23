"""Gold-linked stocks: an intraday book of at most 3 names, driven by the gold price.

2026-11-23: the user asked for a separate section of gold-related stocks whose
trades depend on the gold trend, at most three a day.

Universe (fixed by industry, not by performance): jewellers TITAN, KALYANKJIL,
SENCO, THANGAMAYL, RAJESHEXPO, PCJEWELLER, SKYGOLD, SHANTIGOLD, GOLDIAM, TBZ,
VAIBHAVGBL, BLUESTONE, MOTISONS, PNGJL; gold lenders MUTHOOTFIN, MANAPPURAM,
IIFL.  A name is tradable on day t if, as of t-1: 20-session mean traded value
>= Rs 10 cr, price >= 50, and no trade-for-trade (BE) day in the last 20.

Gold: Dukascopy 1-minute spot XAUUSD.  Known at entry (09:00 IST pre-open):
  g_on    gold 15:30 IST t-1 -> 09:00 IST t   (the overnight move)
  g_trK   gold 15:30 IST K sessions ago -> 09:00 IST t   (K = 5, 20)
  g_ma50  gold at 09:00 t vs the mean of the last 50 15:30 prices
Per stock (t-1 data only): beta of its opening gap on g_on over 250 sessions.

Trades as the main book: open -> close (~15:15 exit), stop 0.75 x 14-day ATR,
13 bps round trip; rules that need today's opening print (gap0) enter just
after the open and pay 5 bps more.

Pre-registered rules (each picks <= 3 names; a day with no signal is flat):
  S1 follow_on_T   g_on > +T: long the 3 highest-beta names; g_on < -T: short them
  S2 fade_on_T     the opposite of S1
  S3 trend20       g_tr20 > 0 and g_ma50 > 0: long top-3 beta; both < 0: short them
  S4 resid_rev     resid = gap0 - beta*g_on; |resid| >= 1%: fade the 3 largest |resid|
  S5 resid_trend   S4, only the side that agrees with the gold trend (S3's sign)
  S6 gap1_short    the main book's rule on gold names: short the 3 largest gap1
                   (yesterday's gap) with gap1 > 1%, only when the gold trend is down
  S7 trend_on      S1 at T=0.3% only when the overnight move agrees with the trend
  C4 raw_gap_rev   control for S4: the same fade on the raw gap, gold ignored
  S8 lenders_follow_T  (added after the DEV correlations) MUTHOOTFIN/MANAPPURAM/IIFL with
                   the overnight gold move: long if g_on > T, short if < -T
  S9 lenders_follow_trend  S8 at 0.3%, not against the gold trend
DEV 2016-07 .. 2023-12, HOLD 2024-01 .. 2026-11 (looked at once).  Pass: DEV
day-level t >= 3 with >= 70% of years positive, then HOLD positive.

    PYTHONPATH=src .venv/bin/python research/gold_book_2026_11/gold_lab.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nse_intraday_ai import gold_book as GB
from nse_intraday_ai import nse_bhav

HERE = Path(__file__).resolve().parent
GOLD_M1 = ROOT / "data" / "commodity" / "duka" / "gold_m1.parquet"
UNIVERSE = GB.UNIVERSE
LENDERS = set(GB.LENDERS)
COST, OPEN_EXTRA, STOP_ATR, K = 13.0, 5.0, 0.75, 3
DEV = ("2016-07-01", "2023-12-31")
HOLD = ("2024-01-01", "2026-12-31")


def load(start="2016-01-01", end=None):
    """Bhavcopy frames of the universe and gold at 09:00/15:30 IST from 1-minute spot XAUUSD."""
    end = end or date.today().isoformat()
    b = nse_bhav.load(date.fromisoformat(start), date.fromisoformat(end), series=("EQ", "BE"))
    fr = GB.frames(b)
    g = pd.read_parquet(GOLD_M1, columns=["close"])["close"]
    return fr, GB.gold_marks(g, fr["open"].index)


def build(d) -> dict:
    """The production features (nse_intraday_ai.gold_book.features), tradable only if it opened."""
    f = GB.features(*d)
    f["ok"] = f["ok"] & f["has_open"]
    return f


def trend_sign(f) -> pd.Series:
    up = (f["g_tr20"] > 0) & (f["g_ma50"] > 0)
    dn = (f["g_tr20"] < 0) & (f["g_ma50"] < 0)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=f["g_tr20"].index)


def book(f, picks: pd.DataFrame, side: pd.DataFrame, extra: float = 0.0) -> pd.DataFrame:
    """Trades for boolean `picks` with `side` (+1 long, -1 short); net bps per trade."""
    r = f["long"].where(side > 0, f["short"])
    net = (r - COST - extra).where(picks & f["ok"] & r.notna())
    t = net.stack().dropna().rename("net").reset_index()
    t.columns = ["session", "symbol", "net"]
    return t


def top_k(score: pd.DataFrame, mask: pd.DataFrame, k: int = K) -> pd.DataFrame:
    s = score.where(mask)
    rank = s.rank(axis=1, ascending=False, method="first")
    return rank <= k


def rules(f) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, float]]:
    ok, beta = f["ok"], f["beta"]
    has_beta = beta.notna() & ok
    out = {}
    tsign = trend_sign(f)
    for T in (0.003, 0.005, 0.01):
        on = f["g_on"]
        sig = pd.Series(np.where(on > T, 1, np.where(on < -T, -1, 0)), index=on.index)
        pick = top_k(beta, has_beta & (beta > 0)).mul(sig != 0, axis=0)
        side = pd.DataFrame(np.repeat(sig.to_numpy()[:, None], beta.shape[1], 1), index=beta.index,
                            columns=beta.columns)
        out[f"S1_follow_on_{T * 100:.1f}"] = (pick, side, 0.0)
        out[f"S2_fade_on_{T * 100:.1f}"] = (pick, -side, 0.0)
    side_tr = pd.DataFrame(np.repeat(tsign.to_numpy()[:, None], beta.shape[1], 1), index=beta.index,
                           columns=beta.columns)
    out["S3_trend20"] = (top_k(beta, has_beta & (beta > 0)).mul(tsign != 0, axis=0), side_tr, 0.0)
    resid = f["gap0"] - beta.mul(f["g_on"], axis=0)
    big = has_beta & (resid.abs() >= 0.01)
    pick4 = top_k(resid.abs(), big)
    side4 = -np.sign(resid)
    out["S4_resid_rev"] = (pick4, side4, OPEN_EXTRA)
    # control: the same fade on the raw gap, gold ignored (does the gold adjustment add anything?)
    rawbig = has_beta & (f["gap0"].abs() >= 0.01)
    out["C4_raw_gap_rev"] = (top_k(f["gap0"].abs(), rawbig), -np.sign(f["gap0"]), OPEN_EXTRA)
    agree = big & (side4.mul(tsign, axis=0) > 0)
    out["S5_resid_trend"] = (top_k(resid.abs(), agree), side4, OPEN_EXTRA)
    m6 = ok & (f["gap1"] > 0.01)
    pick6 = top_k(f["gap1"], m6).mul(tsign < 0, axis=0)
    out["S6_gap1_short_gold_down"] = (pick6, -1 + 0 * beta.fillna(0), 0.0)
    on = f["g_on"]
    sig7 = pd.Series(np.where((on > 0.003) & (tsign > 0), 1, np.where((on < -0.003) & (tsign < 0), -1, 0)),
                     index=on.index)
    side7 = pd.DataFrame(np.repeat(sig7.to_numpy()[:, None], beta.shape[1], 1), index=beta.index,
                         columns=beta.columns)
    out["S7_trend_on"] = (top_k(beta, has_beta & (beta > 0)).mul(sig7 != 0, axis=0), side7, 0.0)
    # S8 (added after the DEV correlations: lenders' *intraday* move tracks the overnight
    # gold move, +0.12; jewellers' does not): follow gold with the gold lenders only.
    lender = pd.DataFrame(False, index=beta.index, columns=beta.columns)
    for s_ in LENDERS & set(beta.columns):
        lender[s_] = True
    for T in (0.0, 0.003, 0.005):
        sig8 = pd.Series(np.where(on > T, 1, np.where(on < -T, -1, 0)), index=on.index)
        side8 = pd.DataFrame(np.repeat(sig8.to_numpy()[:, None], beta.shape[1], 1), index=beta.index,
                             columns=beta.columns)
        out[f"S8_lenders_follow_{T * 100:.1f}"] = ((lender & ok).mul(sig8 != 0, axis=0), side8, 0.0)
    sig9 = pd.Series(np.where((on > 0.003) & (tsign >= 0), 1, np.where((on < -0.003) & (tsign <= 0), -1, 0)),
                     index=on.index)
    side9 = pd.DataFrame(np.repeat(sig9.to_numpy()[:, None], beta.shape[1], 1), index=beta.index,
                         columns=beta.columns)
    out["S9_lenders_follow_trend"] = ((lender & ok).mul(sig9 != 0, axis=0), side9, 0.0)
    return out


def stats(t: pd.DataFrame, lo: str, hi: str) -> dict:
    x = t[(t["session"] >= lo) & (t["session"] <= hi)]
    if len(x) < 10:
        return {"trades": len(x)}
    day = x.groupby("session")["net"].sum() / K                # bps of the 3-slot book's capital
    yr = x.groupby(x["session"].dt.year)["net"].mean()
    return {"trades": len(x), "days": len(day),
            "net_bps_per_trade": round(float(x["net"].mean()), 1),
            "win_pct": round(float((x["net"] > 0).mean() * 100), 1),
            "t_day": round(float(day.mean() / day.std() * np.sqrt(len(day))), 2),
            "years_positive_pct": round(float((yr > 0).mean() * 100), 0),
            "by_year": {int(k): round(float(v), 1) for k, v in yr.items()}}


def main(show_hold: bool) -> None:
    d = load()
    f = build(d)
    res = {"universe_days": int(f["ok"].sum().sum()),
           "names_per_day": round(float(f["ok"].sum(axis=1)[f["ok"].sum(axis=1) > 0].mean()), 1)}
    print(res, flush=True)
    # does the overnight gold move show up in the stocks' gaps and intraday moves at all?
    on = f["g_on"]
    corr = {}
    for s in UNIVERSE:
        if s in f["gap0"]:
            j = pd.concat([on, f["gap0"][s], f["long"][s]], axis=1, keys=["on", "gap", "intra"]).dropna()
            j = j[(j.index >= DEV[0]) & (j.index <= DEV[1])]
            if len(j) > 250:
                corr[s] = {"n": len(j), "gap_vs_gold_on": round(float(j["on"].corr(j["gap"])), 3),
                           "intraday_vs_gold_on": round(float(j["on"].corr(j["intra"])), 3)}
    res["dev_correlations"] = corr
    for name, (pick, side, extra) in rules(f).items():
        t = book(f, pick.astype(bool), side, extra)
        res[name] = {"dev": stats(t, *DEV)}
        if show_hold:
            res[name]["hold"] = stats(t, *HOLD)
        print(name, res[name], flush=True)
    (HERE / ("gold_lab_hold.json" if show_hold else "gold_lab_dev.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main("--hold" in sys.argv)
