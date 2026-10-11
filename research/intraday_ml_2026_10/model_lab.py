"""Walk-forward cross-sectional ML ranker on the hourly panel (no look-ahead).

Monthly expanding refits, embargo of 1 session between the last training session and the
first scored session.  Training rows = liquid names (prior-20d turnover rank <= 300).
Target = forward return (bps) from the decision entry, demeaned per (session, decision bar)
and clipped.  Models: HistGradientBoostingRegressor, Ridge baseline, and an HGB trained on
labels shuffled within each cross-section (control: must give ~0 spread).

Run (heavy):
systemd-run --user --unit=intraday-ml --working-directory=/home/shine/trading-workspace \
  --setenv=PYTHONPATH=src -p Nice=10 -p MemoryMax=6G \
  .venv/bin/python research/intraday_ml_2026_10/model_lab.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel import CACHE, FEATURES, HERE, load_panel  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
PRED = CACHE / "preds.parquet"
LIQ = 300
MIN_TRAIN_SESSIONS = 120
EMBARGO = 1
CLIP = {"eod": 500.0, "1h": 300.0}


def hgb(seed=0):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=500, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)


def demeaned_target(df, hz):
    y = df[f"y_{hz}"].astype(float)
    y = y - y.groupby([df.d, df.b]).transform("mean")
    return y.clip(-CLIP[hz], CLIP[hz])


def fit_predict(kind, tr, te, hz, rng):
    y = demeaned_target(tr, hz).to_numpy()
    if kind == "hgb_shuffle":
        # permute labels within each (session, bar) cross-section: keeps the target's
        # distribution and timing, destroys any feature -> label link
        y = y.copy()
        for idx in tr.groupby(["d", "b"]).indices.values():
            y[idx] = y[rng.permutation(idx)]
    X, Xt = tr[FEATURES].to_numpy(np.float32), te[FEATURES].to_numpy(np.float32)
    if kind in ("hgb", "hgb_shuffle"):
        m = hgb().fit(X, y)
        return m.predict(Xt)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    z = np.nan_to_num(np.clip((X - mu) / sd, -5, 5))
    zt = np.nan_to_num(np.clip((Xt - mu) / sd, -5, 5))
    return Ridge(alpha=10.0).fit(z, y).predict(zt)


def main():
    t0 = time.time()
    panel = load_panel()
    panel = panel[panel.turn_rank <= LIQ].reset_index(drop=True)
    days = np.array(sorted(panel.day.unique()))
    dpos = {d: i for i, d in enumerate(days)}
    panel["di"] = panel.day.map(dpos).astype(int)
    months = sorted({d[:7] for d in days})
    first_of = {m: min(i for i, d in enumerate(days) if d[:7] == m) for m in months}
    oos_months = [m for m in months if first_of[m] - EMBARGO >= MIN_TRAIN_SESSIONS + 20]
    print(f"panel {len(panel)} liquid rows, {len(days)} sessions {days[0]}..{days[-1]}; "
          f"OOS {oos_months[0]}..{oos_months[-1]} ({len(oos_months)} refits)", flush=True)
    kinds = [k for k in sys.argv[1:] if not k.startswith("-")] or ["hgb", "ridge", "hgb_shuffle"]
    preds = pd.read_parquet(PRED) if PRED.exists() else None
    keys = ["d", "s", "b"]
    out = []
    rng = np.random.default_rng(12345)
    importances = {}
    for m in oos_months:
        start = first_of[m]
        # 20-session warm-up for the relative-volume baselines
        tr = panel[(panel.di >= 20) & (panel.di < start - EMBARGO)]
        te = panel[panel.day.str[:7] == m]
        res = te[keys].copy()
        for hz in ("eod", "1h"):
            for kind in kinds:
                res[f"p_{kind}_{hz}"] = fit_predict(kind, tr, te, hz, rng).astype(np.float32)
        out.append(res)
        print(f"{m}: train {len(tr)} rows ({tr.day.nunique()} sessions), test {len(te)}  [{time.time()-t0:.0f}s]", flush=True)
    new = pd.concat(out, ignore_index=True)
    if preds is not None:
        keep = [c for c in preds.columns if c not in new.columns or c in keys]
        new = preds[keep].merge(new, on=keys, how="outer")
    new.to_parquet(PRED, compression="zstd")
    # permutation-free importance proxy on the last fit: correlation of each feature with the
    # final OOS HGB score (diagnostic only)
    if "hgb" in kinds:
        last = panel[panel.day.str[:7] == oos_months[-1]].merge(new, on=keys)
        for hz in ("eod", "1h"):
            importances[hz] = {f: round(float(last[f].corr(last[f"p_hgb_{hz}"], method="spearman")), 3)
                               for f in FEATURES}
        (HERE / "score_feature_corr.json").write_text(json.dumps(importances, indent=1))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
