"""Combine the two gold rules that held up in DEV and HOLD into one <=3-name book.

S8 (lenders follow a >=0.5% overnight gold move) has priority; the slots left are
filled by S4 (fade the gold-adjusted opening gap, |resid| >= 1%) on other names.
Entry for all: just after the open (the list needs today's opening prices), so
every trade pays 13 + 5 bps.  Chosen after the HOLD look -> judged on paper only.

    PYTHONPATH=src .venv/bin/python research/gold_book_2026_11/combine.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gold_lab as L

from nse_intraday_ai import gold_book as GB


def combined(f):
    """The production selection (nse_intraday_ai.gold_book.select), entered at the open."""
    p_l, p_g, side = GB.select(f, stage="open")
    r = f["long"].where(side > 0, f["short"])
    tag = pd.DataFrame(np.where(p_l, "S8", np.where(p_g, "S4", "")), index=side.index, columns=side.columns)
    net = (r - L.COST - L.OPEN_EXTRA).where((p_l | p_g) & r.notna())
    t = net.stack().dropna().rename("net").reset_index()
    t.columns = ["session", "symbol", "net"]
    t["rule"] = [tag.at[a, b] for a, b in zip(t["session"], t["symbol"])]
    return t


def main():
    f = L.build(L.load())
    t = combined(f)
    res = {}
    for name, (lo, hi) in (("dev", L.DEV), ("hold", L.HOLD)):
        res[name] = L.stats(t, lo, hi)
        x = t[(t["session"] >= lo) & (t["session"] <= hi)]
        res[name]["by_rule"] = x.groupby("rule")["net"].agg(["count", "mean"]).round(1).to_dict("index")
        print(name, res[name])
    full = L.stats(t, "2016-07-01", "2026-12-31")
    res["all"] = full
    day = t.groupby("session")["net"].sum() / L.K
    res["last_60_trades"] = round(float(t.tail(60)["net"].mean()), 1)
    res["max_drawdown_pct_of_book"] = round(float(((day / 100).cumsum().cummax() - (day / 100).cumsum()).max()), 2)
    print(full, res["max_drawdown_pct_of_book"])
    (HERE / "combine.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
