"""Look-ahead / alignment audit of the cached hourly panel (panel.py) before trusting any result.

1. Bar alignment: on the 5-minute overlap, the hourly bar starting at HH:15 must close at the
   5m close of the HH+1:10 bar (Yahoo ts = bar START) and open at the HH:15 5m open.
2. Recompute features/targets for random panel rows straight from candles.sqlite3 and the
   bhavcopy (t-1 only) and compare with the cached values.
   Recomputation uses only bars 0..b of the session (features), bar b+1 open (entry) and the
   last bhavcopy session strictly before the day (prior-day features).
3. Pooled univariate Spearman of raw features with the EOD target: a leaked feature would
   show |IC| far above the ~0.01-0.05 typical of genuine intraday predictors.

Run: PYTHONPATH=src .venv/bin/python research/intraday_ml_2026_10/audit_lab.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel import DB, HERE, load_panel  # noqa: E402

from nse_intraday_ai import nse_bhav  # noqa: E402

OUT = HERE / "audit_lab.json"


def bars(con, interval, sym, day):
    lo = f"{day}T00:00:00+00:00"
    hi = f"{day}T23:59:59+00:00"
    df = pd.read_sql("select ts, open, high, low, close, volume from candles where interval=? and symbol=? "
                     "and ts between ? and ? order by ts", con, params=(interval, sym, lo, hi))
    df["t"] = pd.to_datetime(df.ts, utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def alignment(con, rng, n=40):
    syms = [r[0] for r in con.execute("select distinct symbol from candles where interval='5m' limit 400")]
    days = [r[0][:10] for r in con.execute(
        "select distinct substr(ts,1,10) from candles where interval='5m' and ts >= '2026-07-01' order by 1")]
    errs_c, errs_o, n_ok = [], [], 0
    for _ in range(n):
        s, d = syms[rng.integers(len(syms))], days[rng.integers(len(days))]
        h, m = bars(con, "1h", s, d), bars(con, "5m", s, d)
        if h.empty or m.empty:
            continue
        m = m.set_index(m.t.dt.strftime("%H:%M"))
        m = m[~m.index.duplicated(keep="last")]
        for _, r in h.iterrows():
            hh = r.t.hour
            if r.t.minute != 15 or hh > 14:
                continue
            last = f"{hh + 1:02d}:10"
            first = f"{hh:02d}:15"
            if last in m.index and first in m.index:
                errs_c.append(abs(r.close / m.loc[last, "close"] - 1) * 1e4)
                errs_o.append(abs(r.open / m.loc[first, "open"] - 1) * 1e4)
                n_ok += 1
    return {"pairs": n_ok,
            "close_vs_5m_last_bar_bps_median": round(float(np.median(errs_c)), 2),
            "close_match_lt1bp": round(float(np.mean(np.array(errs_c) < 1)), 3),
            "open_vs_5m_first_bar_bps_median": round(float(np.median(errs_o)), 2),
            "open_match_lt1bp": round(float(np.mean(np.array(errs_o) < 1)), 3)}


def recompute(con, panel, symbols_by_s, rng, n=60):
    rows = panel.sample(n, random_state=int(rng.integers(1 << 30)))
    rep = []
    bh_cache = {}
    for _, r in rows.iterrows():
        sym, day, b = symbols_by_s[int(r.s)], r.day, int(r.b)
        h = bars(con, "1h", sym, day)
        h = h[h.t.dt.minute == 15].copy()
        h["bar"] = h.t.dt.hour - 9
        h = h.set_index("bar")
        r_open = h.loc[b, "close"] / h.loc[0, "open"] - 1
        entry = h.loc[b + 1, "open"]
        exit_px = h.loc[[j for j in range(b + 1, 6) if j in h.index], "close"].dropna().iloc[-1]
        y_eod = (exit_px / entry - 1) * 1e4
        y_1h = (h.loc[b + 1, "close"] / entry - 1) * 1e4
        hi = h.loc[0:b, "high"].max()
        lo = h.loc[0:b, "low"].min()
        rp = (h.loc[b, "close"] - lo) / (hi - lo)
        # previous session's official return (bhav of the last session strictly before today)
        d0 = date.fromisoformat(day)
        if d0 not in bh_cache:
            bh = nse_bhav.load(d0 - timedelta(days=10), d0 - timedelta(days=1),
                               columns=["symbol", "close", "prev_close"])
            bh["session"] = pd.to_datetime(bh["session"])
            bh_cache[d0] = bh[bh.session == bh.session.max()].set_index("symbol")
        bh = bh_cache[d0]
        base = sym[:-3]
        prev_ret = (bh.loc[base, "close"] / bh.loc[base, "prev_close"] - 1) if base in bh.index else np.nan
        rep.append({"r_open": abs(r_open - r.r_open), "entry": abs(entry / r.entry - 1),
                    "y_eod": abs(y_eod - r.y_eod), "y_1h": abs(y_1h - r.y_1h),
                    "range_pos": abs(rp - r.range_pos) if np.isfinite(rp) else 0.0,
                    "prev_ret": abs(prev_ret - r.prev_ret) if np.isfinite(prev_ret) else 0.0})
    df = pd.DataFrame(rep)
    return {"rows": len(df), "max_abs_err": {k: float(df[k].max()) for k in df},
            "all_match": bool((df.max() < np.array([1e-4, 1e-4, 0.05, 0.05, 1e-4, 1e-4])).all())}


def main():
    rng = np.random.default_rng(7)
    con = sqlite3.connect(DB)
    panel = load_panel()
    uni = pd.read_csv(HERE.parents[1] / "data/nifty500_symbols.csv")
    syms = sorted(set(r[0] for r in con.execute("select distinct symbol from candles where interval='1h'"))
                  & set(uni.Symbol.astype(str).str.strip() + ".NS"))
    by_s = dict(enumerate(syms))
    out = {"alignment_1h_vs_5m": alignment(con, rng), "recompute": recompute(con, panel, by_s, rng)}
    # univariate ICs of raw features vs EOD target: a leaked feature would show |IC| >> 0.1
    ics = {}
    sub = panel.sample(300_000, random_state=1)
    for f in ("r_open", "r_last1", "range_pos", "dist_vwap", "relvol_cum", "gap", "prev_ret", "ret5"):
        ics[f] = round(float(sub[f].corr(sub.y_eod, method="spearman")), 4)
    out["pooled_spearman_vs_y_eod"] = ics
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
