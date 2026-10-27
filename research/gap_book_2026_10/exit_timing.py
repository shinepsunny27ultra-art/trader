"""When should the gap-reversal short be covered?  Exit-time study on hourly bars.

Picks: the production ranker's walk-forward top-8 (compare_rankers.py "morning" scores,
2019-01 onward, ETFs out, T2T out, top-300).  Path: Yahoo 1-hour bars (2023-10 onward,
current NIFTY-500 names, so ~2/3 of picks are covered).  Entry: the first hourly bar's
open; stop: entry + 0.75 x ATR14 checked on every bar's high (a gap through fills at the
stop, optimistic by the same amount for every exit); exit at the close of the bar
ending at each candidate time.  13 bps per trade.  Paired by session against 15:15.

    PYTHONPATH=src .venv/bin/python research/gap_book_2026_10/exit_timing.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai import ranker as R

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CMP = ROOT / "data" / "ranker_comparison"
EXITS = ["10:15", "11:15", "12:15", "13:15", "14:15", "15:15", "15:30"]
COST = 13.0


def main() -> None:
    panel = pd.read_parquet(CMP / "panel_v2.parquet", columns=["turn_rank", "was_be_20", "atr_pct", "short_bps"])
    score = pd.read_parquet(CMP / "morning_scores.parquet")["score"].reindex(panel.index)
    ok = score.notna() & (panel["turn_rank"] <= 300) & (panel["was_be_20"].fillna(0) == 0)
    s = score[ok]
    rank = s.groupby(level="session").rank(ascending=False, method="first")
    top = s[rank <= 8].index
    top = top[pd.to_datetime(top.get_level_values("session")) >= "2023-11-01"]

    con = sqlite3.connect(f"file:{ROOT / 'data' / 'candles.sqlite3'}?mode=ro", uri=True)
    hb = pd.read_sql("SELECT symbol, ts, open, high, close FROM candles WHERE interval='1h'", con)
    hb["t"] = pd.to_datetime(hb["ts"], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    hb["d"] = hb["t"].dt.date
    hb["sym"] = hb["symbol"].str.removesuffix(".NS")
    hb = hb.sort_values("t").drop_duplicates(["sym", "t"], keep="last")
    groups = dict(iter(hb.groupby(["d", "sym"])))

    rows = []
    for sess, sym in top:
        g = groups.get((sess, sym))
        if g is None or len(g) < 7 or g["t"].iloc[0].strftime("%H:%M") != "09:15":
            continue
        entry = float(g["open"].iloc[0])
        atr = float(panel.at[(sess, sym), "atr_pct"])
        if not (entry > 0 and atr > 0):
            continue
        stop = entry * (1 + 0.75 * atr)
        rec, stopped = {"session": sess, "symbol": sym}, None
        for end, (_, bar) in zip(EXITS, g.iloc[:len(EXITS)].iterrows()):
            if stopped is None and bar["high"] >= stop:
                stopped = -0.75 * atr * 1e4
            rec[end] = stopped if stopped is not None else (entry - bar["close"]) / entry * 1e4
        if all(e in rec for e in EXITS):
            rows.append(rec)
    tr = pd.DataFrame(rows).set_index(["session", "symbol"])
    daily = (tr - COST).groupby(level="session").mean()
    yr = pd.to_datetime(daily.index).year
    out = {
        "trades": int(len(tr)), "sessions": int(len(daily)), "coverage_of_picks": round(len(tr) / len(top), 3),
        "daily_label_check_bps": round(float((panel.loc[tr.index, "short_bps"] - COST).mean()), 1),
        "mean_net_bps": {e: round(float(daily[e].mean()), 1) for e in EXITS},
        "vs_15:15": {e: R.paired(daily[e], daily["15:15"]) for e in EXITS if e != "15:15"},
        "by_year": {e: daily[e].groupby(yr).mean().round(1).to_dict() for e in ("12:15", "14:15", "15:15")},
    }
    (HERE / "exit_timing.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
