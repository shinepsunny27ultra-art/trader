"""Two robustness questions about the 08:15 ranker, same folds as pipeline.weekly
(last 2 years, quarterly refits), booked as traded live (top-8 of 300, T2T out, at the
open, 13 bps; the rule through the pre-open limit):

1. Evening list: the laptop is often off in the morning.  A list built the evening
   before sees US/European/commodity/FX closes and the ADRs one session older.  Does
   it still beat the cloud fallback's fresh gap-rule list?
2. A data source missing at 08:15 (F&O, delivery, global closes, participant OI):
   is the ranker with those features blank still better than the rule?

    PYTHONPATH=src .venv/bin/python research/gap_book_2026_10/robustness.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_rankers import PANEL, VARIANTS  # noqa: E402

from nse_intraday_ai import features as FT  # noqa: E402
from nse_intraday_ai import ranker as R  # noqa: E402
from nse_intraday_ai.gap_reversal import GapReversalConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
US_EU = tuple(p + "_" for p in ("spx", "ndx", "rut", "usvix", "dax", "ftse", "eem", "inda", "dxy", "inr",
                                "crude", "gold", "copper", "us10y"))


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    cfg = VARIANTS["morning"]
    feats = R.feature_columns(panel, cfg)
    mkt = tuple(v + "_" for v in FT.MARKET_SERIES.values())
    groups = {
        "global": [c for c in feats if c.startswith(mkt) or c.startswith("adr")],
        "fo": [c for c in feats if c.startswith(("fo_", "fut", "opt", "basis", "pcr", "call", "put"))],
        "participant": [c for c in feats if c.startswith("oi_")],
        "delivery": [c for c in feats if "deliv" in c],
    }
    evening = [c for c in feats if c.startswith(US_EU) or c.startswith("adr")]
    print({k: len(v) for k, v in groups.items()}, "evening-shifted:", len(evening), flush=True)
    shifted = panel[evening].groupby(level="symbol").shift(1)
    d = pd.to_datetime(panel.index.get_level_values("session"))
    last = d.max()
    start = last - pd.DateOffset(years=2)
    edges = pd.DatetimeIndex([start]).append(pd.date_range(start, last + pd.Timedelta(days=1), freq="QS"))
    edges = edges.append(pd.DatetimeIndex([last + pd.Timedelta(days=1)])).unique().sort_values()
    variants = ["normal", "evening"] + [f"blank_{g}" for g in groups]
    out = {v: pd.Series(np.nan, index=panel.index) for v in variants}
    for a, b in zip(edges[:-1], edges[1:]):
        test = (d >= a) & (d < b)
        if not test.any():
            continue
        model = R.Ranker.fit(panel[d < a], cfg)
        T = panel[test]
        out["normal"][test] = model.score(T)
        out["evening"][test] = model.score(T.assign(**{c: shifted.loc[test, c] for c in evening}))
        for g, cols in groups.items():
            out[f"blank_{g}"][test] = model.score(T.assign(**{c: np.nan for c in cols}))
        print(f"fold {a.date()} done {time.time() - t0:.0f}s", flush=True)
    clean = panel["was_be_20"].fillna(0) == 0
    books = {v: R.book(panel, s.where(clean), universe=300) for v, s in out.items()}
    books["rule"] = R.book(panel, panel["gap1"].where(clean), universe=300,
                           entry_limit=GapReversalConfig().entry_limit_pct)
    s0 = str(start.date())
    res = {"window": [s0, str(last.date())],
           "summary": {k: R.summarize(v, s0) for k, v in books.items()},
           "vs_normal": {k: R.paired(v, books["normal"], s0) for k, v in books.items() if k != "normal"},
           "vs_rule": {k: R.paired(v, books["rule"], s0) for k, v in books.items() if k != "rule"},
           "top8_overlap_evening_vs_normal": None}
    top = {}
    for v in ("normal", "evening"):
        s = out[v].where(clean & (panel["turn_rank"] <= 300)).dropna()
        top[v] = set(s.groupby(level="session", group_keys=False).nlargest(8).index)
    res["top8_overlap_evening_vs_normal"] = round(len(top["normal"] & top["evening"]) / len(top["normal"]), 3)
    (HERE / "robustness.json").write_text(json.dumps(res, indent=1, default=str))
    for k in books:
        s = res["summary"][k]
        n, r = res["vs_normal"].get(k, {}), res["vs_rule"].get(k, {})
        print(f"{k:18s} {s['net_bps_per_trade']:7.2f} bps t {s['t_stat']:5.2f} | vs normal "
              f"{n.get('mean_diff_bps', 0):+6.2f} (t {n.get('t_stat', 0):+5.2f}) | vs rule "
              f"{r.get('mean_diff_bps', 0):+6.2f} (t {r.get('t_stat', 0):+5.2f})")
    print("evening top-8 overlap with the 08:15 list:", res["top8_overlap_evening_vs_normal"],
          f"| total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
