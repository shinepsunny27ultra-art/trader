"""Evaluate the walk-forward ML scores as tradable books, net of the 13.3 bps MIS round trip.

For every (model, horizon, k, stop, leg) variant: per-trade net bps, per-session mean, t,
hit rate, worst day, max drawdown, by year / quarter / decision bar / leg.  Holm over the
whole family (pooled variants AND per-decision-time slices) and a deflated Sharpe that
charges for every trial, including the ~100 rule variants tested earlier today.

Run: PYTHONPATH=src .venv/bin/python research/intraday_ml_2026_10/eval_lab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_lab import PRED  # noqa: E402
from panel import COST_BPS, HERE, load_panel  # noqa: E402

from nse_intraday_ai.ranker import deflated_sharpe, holm  # noqa: E402

OUT = HERE / "eval_lab.json"
PRIOR_TRIALS = 100            # rule-based intraday variants tested earlier on 2026-10-05
MODELS = ("hgb", "ridge")
CONTROLS = ("hgb_shuffle",)
KS = (1, 5, 10)
STOPS = (None, 0.5, 1.0)
ND = NormalDist()


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def pnl_col(df, side, hz, stop):
    if stop is None:
        return side * df[f"y_{hz}"].astype(float)
    return df[f"pnl_{'L' if side > 0 else 'S'}_{hz}_s{stop}"].astype(float)


def trades_for(df, score, hz, k, stop, leg):
    """Rows traded by one variant, with a `net` column (bps after costs)."""
    g = df.groupby(["d", "b"])[score]
    rk_hi = g.rank(ascending=False, method="first")
    rk_lo = g.rank(ascending=True, method="first")
    parts = []
    if leg in ("L", "LS"):
        t = df[rk_hi <= k].copy()
        t["side"] = 1
        parts.append(t)
    if leg in ("S", "LS"):
        t = df[rk_lo <= k].copy()
        t["side"] = -1
        parts.append(t)
    if leg == "best1":
        med = g.transform("median")
        dev = df[score] - med
        hi = df[rk_hi <= 1].assign(side=1, conv=dev[rk_hi <= 1])
        lo = df[rk_lo <= 1].assign(side=-1, conv=-dev[rk_lo <= 1])
        both = pd.concat([hi, lo]).sort_values("conv", ascending=False)
        parts.append(both.drop_duplicates(["d", "b"]).drop(columns="conv"))
    t = pd.concat(parts)
    t["gross"] = np.where(t.side > 0, pnl_col(t, 1, hz, stop), pnl_col(t, -1, hz, stop))
    t["net"] = t.gross - COST_BPS
    return t


def summarize(t, detail=True):
    daily = t.groupby("day").net.mean()
    cum = daily.cumsum()
    r = {"trades": int(len(t)), "sessions": int(len(daily)),
         "net": round(float(t.net.mean()), 2), "gross": round(float(t.gross.mean()), 2),
         "t": round(tstat(daily), 2), "hit": round(float((t.net > 0).mean()), 3),
         "worst_day": round(float(daily.min()), 1), "best_day": round(float(daily.max()), 1),
         "max_dd": round(float((cum - cum.cummax()).min()), 1),
         "sharpe_ann": round(float(daily.mean() / daily.std(ddof=1) * np.sqrt(250)), 2) if len(daily) > 2 else 0.0}
    if detail:
        r["by_year"] = {y: round(float(x.mean()), 2) for y, x in t.groupby(t.day.str[:4]).net}
        r["by_quarter"] = {q: [round(float(x.net.mean()), 2), round(tstat(x.groupby("day").net.mean()), 2)]
                           for q, x in t.groupby(t.day.str[:4] + "Q" + ((t.day.str[5:7].astype(int) - 1) // 3 + 1).astype(str))}
        r["by_bar"] = {int(b): round(float(x.mean()), 2) for b, x in t.groupby("b").net}
        r["by_leg"] = {("long" if s > 0 else "short"): round(float(x.mean()), 2) for s, x in t.groupby("side").net}
        top = daily.sort_values(ascending=False)
        r["share_top10_days"] = round(float(top.head(10).sum() / daily.sum()), 2) if daily.sum() != 0 else None
        r["net_ex_top10_days"] = round(float(top.iloc[10:].mean()), 2)
        r["median_day"] = round(float(daily.median()), 2)
    return r, daily


def ic_report(df, score, hz):
    y = df[f"y_{hz}"]
    ic = df.assign(_y=y).groupby(["d", "b"]).apply(
        lambda g: g[score].corr(g._y, method="spearman"), include_groups=False)
    by_day = ic.groupby(level=0).mean()
    dec = pd.qcut(df.groupby(["d", "b"])[score].rank(pct=True), 10, labels=False)
    spread = df.assign(dec=dec).groupby("dec")[f"y_{hz}"].mean()
    return {"ic_mean": round(float(ic.mean()), 4), "ic_t": round(tstat(by_day), 2),
            "decile_gross_bps": [round(float(x), 1) for x in spread]}


def main():
    panel = load_panel()
    preds = pd.read_parquet(PRED)
    df = panel.merge(preds, on=["d", "s", "b"], how="inner")
    out = {"oos": [df.day.min(), df.day.max()], "sessions": int(df.day.nunique()), "cost_bps": COST_BPS,
           "ic": {}, "variants": {}, "controls": {}, "slices": {}}
    pvals, sharpes, dailies = {}, [], {}
    for model in MODELS + CONTROLS:
        for hz in ("eod", "1h"):
            score = f"p_{model}_{hz}"
            if score not in df:
                continue
            out["ic"][f"{model}_{hz}"] = ic_report(df, score, hz)
            print(model, hz, out["ic"][f"{model}_{hz}"], flush=True)
            for k in KS:
                legs = ("L", "S", "LS") + (("best1",) if k == 1 else ())
                for stop in STOPS:
                    for leg in legs:
                        name = f"{model}_{hz}_k{k}_{leg}_stop{stop or 0}"
                        t = trades_for(df, score, hz, k, stop, leg)
                        r, daily = summarize(t, detail=model not in CONTROLS)
                        if model in CONTROLS:
                            out["controls"][name] = r
                            continue
                        out["variants"][name] = r
                        pvals[name] = 1 - ND.cdf(r["t"])
                        sharpes.append(float(daily.mean() / daily.std(ddof=1)))
                        dailies[name] = daily
                        for b, tb in t.groupby("b"):
                            rb, db = summarize(tb, detail=False)
                            out["slices"][f"{name}@b{int(b)}"] = rb
                            pvals[f"{name}@b{int(b)}"] = 1 - ND.cdf(rb["t"])
                            sharpes.append(float(db.mean() / db.std(ddof=1)) if len(db) > 2 else 0.0)
    n_trials = len(pvals)
    adj = holm(pvals)
    for name, r in list(out["variants"].items()) + list(out["slices"].items()):
        r["holm_p"] = adj[name]
    best = max(out["variants"], key=lambda n: out["variants"][n]["t"])
    best_slice = max(out["slices"], key=lambda n: out["slices"][n]["t"])
    out["n_trials_this_study"] = n_trials
    out["n_trials_total_for_dsr"] = n_trials + PRIOR_TRIALS
    out["best_variant"] = best
    out["best_variant_dsr"] = deflated_sharpe(dailies[best], n_trials + PRIOR_TRIALS, sharpes)
    out["best_slice"] = best_slice
    out["n_positive_net"] = int(sum(r["net"] > 0 for r in out["variants"].values()))
    out["n_holm_significant"] = int(sum(p < 0.05 for p in adj.values()))
    OUT.write_text(json.dumps(out, indent=1))
    rows = sorted(out["variants"].items(), key=lambda kv: -kv[1]["t"])
    print(f"\n{n_trials} trials; best {best}; DSR {out['best_variant_dsr']}")
    print(f"{'variant':<34} {'net':>7} {'gross':>7} {'t':>6} {'hit':>5} {'sess':>5} {'trades':>7} {'worst':>7} {'maxDD':>8} holm")
    for n, r in rows[:25]:
        print(f"{n:<34} {r['net']:+7.2f} {r['gross']:+7.2f} {r['t']:+6.2f} {r['hit']:5.3f} {r['sessions']:5d} "
              f"{r['trades']:7d} {r['worst_day']:+7.1f} {r['max_dd']:+8.0f} {r['holm_p']:.3f}")
    print("\ncontrols (shuffled labels):")
    for n, r in out["controls"].items():
        if "_k10_LS_stop0" in n or "_k5_LS_stop0" in n:
            print(f"{n:<40} net {r['net']:+7.2f} gross {r['gross']:+7.2f} t {r['t']:+6.2f}")
    print("\nbest slices:")
    for n, r in sorted(out["slices"].items(), key=lambda kv: -kv[1]["t"])[:10]:
        print(f"{n:<40} net {r['net']:+7.2f} gross {r['gross']:+7.2f} t {r['t']:+6.2f} n {r['sessions']} holm {r['holm_p']:.3f}")


if __name__ == "__main__":
    main()
