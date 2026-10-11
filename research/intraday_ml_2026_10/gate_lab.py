"""Cost-gated books on the walk-forward ML scores: trade only when the predicted edge clears costs.

eval_lab.py always trades the extremes of every cross-section.  The natural ML trading rule
for "the single best stock right now" is different: act only when the model's predicted
cross-sectional edge (score, in bps of demeaned return) is larger than a threshold, else
stand aside.  Books per decision (session, bar), liquid names only (turnover rank <= 300):

  best1_g : the one name with the largest |score| if |score| >= thr (long if >0, short if <0)
  L5_g    : up to 5 longs with score >= thr          S5_g : up to 5 shorts with score <= -thr

thr in {15, 30, 60} bps; stops 0 / 0.5 / 1.0 ATR (hourly highs/lows, stop first) as in
eval_lab.  Holm is applied over this study's variants PLUS every p-value from eval_lab.json
(pooled variants and per-decision slices), so the family is the whole ML search.
Also reports score calibration (predicted vs realised demeaned bps by score bucket).

Run: PYTHONPATH=src .venv/bin/python research/intraday_ml_2026_10/gate_lab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lab import OUT as EVAL_OUT  # noqa: E402
from eval_lab import PRIOR_TRIALS, pnl_col, summarize  # noqa: E402
from model_lab import PRED  # noqa: E402
from panel import COST_BPS, HERE, load_panel  # noqa: E402

from nse_intraday_ai.ranker import deflated_sharpe, holm  # noqa: E402

OUT = HERE / "gate_lab.json"
MODELS = ("hgb", "ridge")
HZS = ("eod", "1h")
THRS = (15.0, 30.0, 60.0)
STOPS = (None, 0.5, 1.0)
BOOKS = ("best1_g", "L5_g", "S5_g")
MIN_TRADES = 100
ND = NormalDist()


def gated(df, score, thr, book):
    s = df[score]
    if book == "best1_g":
        a = s.abs()
        idx = a.groupby([df.d, df.b]).idxmax()
        t = df.loc[idx.to_numpy()]
        t = t[t[score].abs() >= thr].copy()
        t["side"] = np.sign(t[score]).astype(int)
        return t
    side = 1 if book.startswith("L") else -1
    cand = df[side * s >= thr]
    rk = (side * cand[score]).groupby([cand.d, cand.b]).rank(ascending=False, method="first")
    t = cand[rk <= 5].copy()
    t["side"] = side
    return t


def with_pnl(t, hz, stop):
    t = t.copy()
    t["gross"] = np.where(t.side > 0, pnl_col(t, 1, hz, stop), pnl_col(t, -1, hz, stop))
    t["net"] = t.gross - COST_BPS
    return t


def calibration(df, score, hz):
    y = df[f"y_{hz}"].astype(float)
    ydm = y - y.groupby([df.d, df.b]).transform("mean")
    edges = [-1e9, -60, -30, -15, -5, 5, 15, 30, 60, 1e9]
    bucket = pd.cut(df[score], edges)
    g = pd.DataFrame({"p": df[score], "y": ydm, "bk": bucket}).groupby("bk", observed=True)
    return {str(k): {"n": int(len(v)), "pred": round(float(v.p.mean()), 1), "realised": round(float(v.y.mean()), 1)}
            for k, v in g}


def main():
    panel = load_panel()
    preds = pd.read_parquet(PRED)
    df = panel.merge(preds, on=["d", "s", "b"], how="inner")
    n_sessions = int(df.day.nunique())
    out = {"oos": [df.day.min(), df.day.max()], "sessions": n_sessions, "cost_bps": COST_BPS,
           "calibration": {}, "variants": {}, "controls": {}}
    pvals, sharpes, dailies = {}, [], {}
    for model in MODELS + ("hgb_shuffle",):
        for hz in HZS:
            score = f"p_{model}_{hz}"
            if model != "hgb_shuffle":
                out["calibration"][f"{model}_{hz}"] = calibration(df, score, hz)
            for thr in THRS:
                for book in BOOKS:
                    base = gated(df, score, thr, book)
                    for stop in STOPS:
                        name = f"{model}_{hz}_{book}_thr{int(thr)}_stop{stop or 0}"
                        if len(base) < MIN_TRADES:
                            if model != "hgb_shuffle":
                                pvals[name] = 1.0      # counted as a trial, never selectable
                                out["variants"][name] = {"trades": int(len(base)), "skipped": True}
                            continue
                        t = with_pnl(base, hz, stop)
                        r, daily = summarize(t, detail=model != "hgb_shuffle")
                        r["sessions_traded_share"] = round(r["sessions"] / n_sessions, 3)
                        if model == "hgb_shuffle":
                            out["controls"][name] = r
                            continue
                        out["variants"][name] = r
                        pvals[name] = 1 - ND.cdf(r["t"])
                        sharpes.append(float(daily.mean() / daily.std(ddof=1)) if len(daily) > 2 else 0.0)
                        dailies[name] = daily
            print(model, hz, "done", flush=True)
    # family = this study + every pooled variant and slice from eval_lab
    prev = json.loads(EVAL_OUT.read_text())
    fam = dict(pvals)
    for sect in ("variants", "slices"):
        for n, r in prev[sect].items():
            fam[f"eval:{n}"] = 1 - ND.cdf(r["t"])
    adj = holm(fam)
    for n in pvals:
        out["variants"][n]["holm_p_family"] = adj[n]
    live = {n: r for n, r in out["variants"].items() if not r.get("skipped")}
    best = max(live, key=lambda n: live[n]["t"])
    out["n_trials_this_study"] = len(pvals)
    out["n_trials_family"] = len(fam)
    out["n_trials_total_for_dsr"] = len(fam) + PRIOR_TRIALS
    out["best_variant"] = best
    out["best_variant_dsr"] = deflated_sharpe(dailies[best], len(fam) + PRIOR_TRIALS, sharpes)
    out["n_positive_net"] = int(sum(r["net"] > 0 for r in live.values()))
    out["n_t_ge_3"] = int(sum(r["t"] >= 3 for r in live.values()))
    out["n_holm_significant"] = int(sum(adj[n] < 0.05 for n in pvals))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n{len(pvals)} gated trials (family {len(fam)}); best {best}; DSR {out['best_variant_dsr']}")
    print(f"{'variant':<40} {'net':>7} {'gross':>7} {'t':>6} {'hit':>5} {'sess':>5} {'trades':>7} {'worst':>7} {'maxDD':>8} holm")
    for n, r in sorted(live.items(), key=lambda kv: -kv[1]["t"])[:25]:
        print(f"{n:<40} {r['net']:+7.2f} {r['gross']:+7.2f} {r['t']:+6.2f} {r['hit']:5.3f} {r['sessions']:5d} "
              f"{r['trades']:7d} {r['worst_day']:+7.1f} {r['max_dd']:+8.0f} {r['holm_p_family']:.3f}")
    print("\ncontrols (shuffled labels):")
    for n, r in out["controls"].items():
        if n.endswith("stop0"):
            print(f"{n:<44} net {r['net']:+7.2f} gross {r['gross']:+7.2f} t {r['t']:+6.2f} trades {r['trades']}")
    print("\ncalibration hgb_eod:", json.dumps(out["calibration"]["hgb_eod"]))


if __name__ == "__main__":
    main()
