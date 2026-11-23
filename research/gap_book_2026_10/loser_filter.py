"""Do names that keep losing in the book deserve a blacklist?

2026-11-23: the live paper book showed names picked again and again with losses
(KANOHAR 7 trades -₹11k, OMNI 4/-₹11k, IFCI 3/3 losers, PFOCUS, IXIGO, CYIENT...).
Question: should a name be dropped from the list after a run of losing trades?

Tested walk-forward on the production recipe's out-of-sample scores
(data/ranker_comparison/morning_scores.parquet, 2019-01..2026-10), booked as
traded live (top-8 of 300, T2T out, 13 bps).  Each rule is simulated
session by session: it only knows the book's own trades that closed before the
day it ranks (a dropped name no longer trades, so its record stops updating
until the lookback expires - exactly what live would do).

Families:
  streakN_L   drop a name whose last N book trades (within L sessions) all lost
  lossK_of_M  drop a name that lost on >= K of its last M book trades (within 60)
  cumX        drop a name whose book trades in the last 60 sessions sum < -X bps
                (needs >= 2 trades)
  cand_*      same idea on every session the name was a candidate (not only traded):
                drop if its mean short_bps on the last 20 candidate sessions < -X
Control: `random_drop` drops as many names per day as streak2_60 does, at random,
to show what merely skipping to lower-ranked names costs.

    PYTHONPATH=src .venv/bin/python research/gap_book_2026_10/loser_filter.py
Output: research/gap_book_2026_10/loser_filter.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nse_intraday_ai import ranker as R  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "ranker_comparison"
K, UNIVERSE, COST = 8, 300, 13.0
DEV_END, HOLD_START = "2025-08-31", "2025-09-01"


def load() -> pd.DataFrame:
    p = pd.read_parquet(OUT / "panel_v2.parquet", columns=["short_bps", "turn_rank", "was_be_20"])
    s = pd.read_parquet(OUT / "morning_scores.parquet")["score"]
    df = p.join(s, how="inner")
    df = df[(df["was_be_20"].fillna(0) == 0) & (df["turn_rank"] <= UNIVERSE)
            & df["score"].notna() & df["short_bps"].notna()]
    return df[["score", "short_bps"]]


def simulate(df: pd.DataFrame, rule, rng=None, drop_counts=None) -> tuple[pd.Series, pd.Series]:
    """Daily net bps of the filtered book, and the number of names dropped per day."""
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))   # (session_idx, net) of book trades
    cand: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))   # (session_idx, short_bps) as candidate
    daily, dropped = {}, {}
    for i, (sess, g) in enumerate(df.groupby(level="session", sort=True)):
        g = g.droplevel("session").sort_values("score", ascending=False)
        if rule == "random":
            n = int(drop_counts.get(sess, 0))
            head = g.index[:K + n]
            out = set(rng.choice(head, size=n, replace=False)) if n else set()
        elif rule is None:
            out = set()
        else:
            out = {s for s in g.index[:K + 40] if rule(i, hist.get(s, ()), cand.get(s, ()))}
        top = [s for s in g.index if s not in out][:K]
        dropped[sess] = sum(1 for s in g.index[:K] if s in out)
        net = g.loc[top, "short_bps"] - COST
        daily[sess] = float(net.mean())
        for s, v in net.items():
            hist[s].append((i, float(v)))
        for s, v in g["short_bps"].items():
            cand[s].append((i, float(v)))
    return pd.Series(daily), pd.Series(dropped)


def recent(h, i, lookback):
    return [v for j, v in h if i - j <= lookback]


def streak(n, lookback):
    def f(i, h, c):
        r = recent(h, i, lookback)[-n:]
        return len(r) == n and all(v < 0 for v in r)
    return f


def loss_k_of_m(k, m, lookback=60):
    def f(i, h, c):
        r = recent(h, i, lookback)[-m:]
        return len(r) >= k and sum(v < 0 for v in r) >= k
    return f


def cum_below(x, lookback=60):
    def f(i, h, c):
        r = recent(h, i, lookback)
        return len(r) >= 2 and sum(r) < -x
    return f


def cand_mean_below(x):
    def f(i, h, c):
        r = [v for j, v in c if i - j <= 30]
        return len(r) >= 10 and np.mean(r) < -x
    return f


RULES = {
    "streak2_20": streak(2, 20), "streak2_60": streak(2, 60), "streak3_60": streak(3, 60),
    "streak2_120": streak(2, 120), "streak3_120": streak(3, 120),
    "loss3_of_4": loss_k_of_m(3, 4), "loss4_of_5": loss_k_of_m(4, 5), "loss2_of_3": loss_k_of_m(2, 3),
    "cum300": cum_below(300), "cum500": cum_below(500), "cum800": cum_below(800),
    "cand_mean40": cand_mean_below(40), "cand_mean80": cand_mean_below(80),
}


def main() -> None:
    df = load()
    base, _ = simulate(df, None)
    res = {"base": {"dev": R.summarize(base, end=DEV_END), "hold": R.summarize(base, start=HOLD_START)}}
    drops = {}
    for name, rule in RULES.items():
        d, n = simulate(df, rule)
        drops[name] = n
        res[name] = {
            "dropped_per_day": round(float(n.mean()), 3),
            "dev_vs_base": R.paired(d, base, end=DEV_END),
            "hold_vs_base": R.paired(d, base, start=HOLD_START),
            "all_vs_base": R.paired(d, base),
            "last_12m_vs_base": R.paired(d, base, start=str((pd.Timestamp(base.index.max())
                                                            - pd.DateOffset(years=1)).date())),
        }
        print(name, res[name]["dropped_per_day"], res[name]["dev_vs_base"], res[name]["hold_vs_base"], flush=True)
    rng = np.random.default_rng(7)
    ctrl = []
    for _ in range(20):
        d, _n = simulate(df, "random", rng=rng, drop_counts=drops["streak2_60"])
        ctrl.append(R.paired(d, base)["mean_diff_bps"])
    res["random_drop_like_streak2_60"] = {"mean_diff_bps_all": round(float(np.mean(ctrl)), 2),
                                          "sd": round(float(np.std(ctrl)), 2)}
    print("random", res["random_drop_like_streak2_60"])
    (HERE / "loser_filter.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
