"""Does this morning's Asian open predict the day for the gap-reversal short book?

The book loses on broad intraday rallies (2026-10-27: median liquid stock +96 bps from
the open; 10-29: +85).  At 08:15 IST Tokyo, Seoul, Taipei, Singapore, Hong Kong and
Shanghai have opened; `asia_open0` is the mean of their opening gaps.  Questions:

1. Does `asia_open0` predict the Indian market's open->close move (median of the
   top-300 universe), or the book's net return?  DEV 2019-01..2025-08, HOLD 2025-09..
2. Is there a size-down rule worth testing?  Terciles are cut on DEV only.

Book = production walk-forward scores (compare_rankers "morning"), top-8 of 300,
T2T out, at the open, 13 bps.

    PYTHONPATH=src .venv/bin/python research/gap_book_2026_10/asia_timing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai import ranker as R

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CMP = ROOT / "data" / "ranker_comparison"
DEV_END, HOLD_START = "2025-08-31", "2025-09-01"


def main() -> None:
    cols = ["short_bps", "intra_bps", "turn_rank", "was_be_20", "asia_open0", "n225_open0", "hsi_open0"]
    panel = pd.read_parquet(CMP / "panel_v2.parquet", columns=cols)
    score = pd.read_parquet(CMP / "morning_scores.parquet")["score"].reindex(panel.index)
    book = R.book(panel, score.where(panel["was_be_20"].fillna(0) == 0), universe=300)
    univ = panel[panel["turn_rank"] <= 300]
    day = pd.DataFrame({
        "book": book,
        "market_oc": univ.groupby(level="session")["intra_bps"].median(),
        "asia": univ.groupby(level="session")["asia_open0"].first() * 1e4,
    }).dropna()
    day.index = pd.to_datetime(day.index)
    day = day[day.index >= "2019-01-01"]
    out = {"sessions": int(len(day))}
    for name, part in (("dev", day[day.index <= DEV_END]), ("hold", day[day.index >= HOLD_START])):
        r = {"sessions": int(len(part)),
             "corr_asia_vs_market_oc": round(float(part["asia"].corr(part["market_oc"])), 3),
             "corr_asia_vs_book": round(float(part["asia"].corr(part["book"])), 3),
             "spearman_asia_vs_book": round(float(part["asia"].corr(part["book"], method="spearman")), 3)}
        out[name] = r
    dev = day[day.index <= DEV_END]
    cuts = dev["asia"].quantile([1 / 3, 2 / 3]).to_numpy()
    day["tercile"] = np.digitize(day["asia"], cuts)          # 0 = Asia opened weakest, 2 = strongest
    for name, part in (("dev", day[day.index <= DEV_END]), ("hold", day[day.index >= HOLD_START])):
        g = part.groupby("tercile").agg(days=("book", "size"), book_bps=("book", "mean"),
                                        book_down_pct=("book", lambda x: (x < 0).mean() * 100),
                                        market_oc_bps=("market_oc", "mean"))
        out[f"{name}_by_asia_tercile"] = g.round(1).to_dict("index")
    out["tercile_cuts_bps"] = [round(float(c), 1) for c in cuts]
    (HERE / "asia_timing.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
