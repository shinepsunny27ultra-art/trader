"""Can the day's Indian market direction (open -> close) be predicted at 08:15 IST?

The short book loses on broad intraday rallies (corr(book, market open->close) ~ -0.46).
If the market's open->close move were predictable from what is known at 08:15, the book
could trade bigger on predicted-down days and smaller on predicted-up days.

Target: median open->close (bps) of the day's top-300 liquid universe.
Inputs: every market-level feature of the production panel (one value per session:
global closes, NIFTY/BANK/midcap, India VIX, FII/DII/Pro/Client positioning, breadth,
yesterday's market moves, calendar) — all dated before the session.
Walk-forward: yearly refits, expanding window, predictions from 2019 on.  Models: ridge
on standardised inputs, and a small gradient-boosted tree.  DEV 2019..2025-08, HOLD after.

    PYTHONPATH=src .venv/bin/python research/gap_book_2026_10/market_direction.py
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


def market_frame() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(CMP / "panel_v2.parquet")
    u = panel[panel["turn_rank"] <= 300]
    sess = u.index.get_level_values("session")
    # market-level columns: one value per session (checked on 150 sampled sessions);
    # today's opening-gap family (gap0, x_gap0_*) is not known at 08:15 and is left out
    num = [c for c in u.columns if u[c].dtype.kind == "f" and c not in R.NON_FEATURES]
    days = pd.Index(sorted(set(sess)))
    sample = u[sess.isin(days[:: max(1, len(days) // 150)])]
    nuniq = sample[num].groupby(sample.index.get_level_values("session")).nunique().max()
    mkt_cols = [c for c in num if nuniq[c] <= 1 and "gap0" not in c]
    df = u[mkt_cols].groupby(sess).first()
    df["target"] = u.groupby(sess)["intra_bps"].median()
    score = pd.read_parquet(CMP / "morning_scores.parquet")["score"].reindex(panel.index)
    df["book"] = R.book(panel, score.where(panel["was_be_20"].fillna(0) == 0), universe=300)
    df.index = pd.to_datetime(df.index)
    return df, mkt_cols


def walk_forward(df: pd.DataFrame, cols: list[str], kind: str) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    pred = pd.Series(np.nan, index=df.index)
    for year in range(2019, df.index.max().year + 1):
        train = df[(df.index < f"{year}-01-01") & df["target"].notna()]
        test = df[(df.index >= f"{year}-01-01") & (df.index < f"{year + 1}-01-01")]
        if len(train) < 400 or test.empty:
            continue
        X, Xt = train[cols], test[cols]
        if kind == "ridge":
            mu, sd = X.mean(), X.std().replace(0, 1)
            Xs, Xts = ((X - mu) / sd).fillna(0), ((Xt - mu) / sd).fillna(0)
            m = Ridge(alpha=50.0).fit(Xs, train["target"].clip(-300, 300))
            pred[test.index] = m.predict(Xts)
        else:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_leaf_nodes=8,
                                              min_samples_leaf=60, l2_regularization=5.0, random_state=0)
            m.fit(X, train["target"].clip(-300, 300))
            pred[test.index] = m.predict(Xt)
    return pred


def main() -> None:
    df, cols = market_frame()
    df = df[df.index >= "2016-06-01"]
    out = {"market_features": len(cols), "sessions": int(df["target"].notna().sum())}
    for kind in ("ridge", "hgb"):
        df[f"pred_{kind}"] = walk_forward(df, cols, kind)
    ev = df[df.index >= "2019-01-01"].dropna(subset=["target", "book", "pred_ridge", "pred_hgb"])
    for name, part in (("dev", ev[ev.index <= "2025-08-31"]), ("hold", ev[ev.index >= "2025-09-01"])):
        r = {"sessions": int(len(part)),
             "corr_book_vs_market": round(float(part["book"].corr(part["target"])), 3)}
        for kind in ("ridge", "hgb"):
            p = part[f"pred_{kind}"]
            r[f"{kind}_corr_pred_vs_market"] = round(float(p.corr(part["target"])), 3)
            r[f"{kind}_direction_hit_pct"] = round(float((np.sign(p) == np.sign(part["target"])).mean() * 100), 1)
            q = pd.qcut(p, 3, labels=["predicted down", "middle", "predicted up"])
            g = part.groupby(q, observed=True).agg(market=("target", "mean"), book=("book", "mean"), days=("book", "size"))
            r[f"{kind}_by_prediction_tercile"] = g.round(1).to_dict("index")
            # a sizing overlay: 1.5x on predicted-down tercile, 0.5x on predicted-up, 1x otherwise
            w = q.map({"predicted down": 1.5, "middle": 1.0, "predicted up": 0.5}).astype(float)
            sized = part["book"] * w
            r[f"{kind}_overlay"] = {"base_mean": round(float(part["book"].mean()), 1),
                                    "sized_mean": round(float(sized.mean()), 1),
                                    "base_sharpe": round(float(part["book"].mean() / part["book"].std() * 250 ** 0.5), 2),
                                    "sized_sharpe": round(float(sized.mean() / sized.std() * 250 ** 0.5), 2)}
        out[name] = r
    (HERE / "market_direction.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
