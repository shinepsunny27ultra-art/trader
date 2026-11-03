from __future__ import annotations

import numpy as np
import pandas as pd

from nse_intraday_ai import market_outlook as MO


def _panel(seed: int = 0) -> pd.DataFrame:
    """6 years x 5 stocks; the day's market move follows a market-level input `spx_r1`."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", "2024-12-31").date
    spx = rng.normal(0, 1, len(days))
    rows = []
    for d, x in zip(days, spx):
        for s in range(5):
            rows.append({"session": d, "symbol": f"S{s}", "spx_r1": x, "dow": float(pd.Timestamp(d).dayofweek),
                         "gap1": rng.normal(), "turn_rank": float(s + 1),
                         "intra_bps": 40 * x + rng.normal(0, 20), "short_bps": rng.normal(0, 50)})
    return pd.DataFrame(rows).set_index(["session", "symbol"]).astype(
        {"spx_r1": "float32", "dow": "float32", "gap1": "float32", "turn_rank": "float32"})


def test_market_columns_are_the_per_session_constants():
    cols = MO.market_columns(_panel())
    assert set(cols) == {"spx_r1", "dow"}                     # gap1 varies by stock


def test_outlook_flags_a_rally_and_halves_the_size(tmp_path):
    o = MO.Outlook.fit(_panel())
    assert o.oos["direction_hit"] > 0.8 and o.oos["sessions"] > 400
    up = o.predict(pd.DataFrame({"spx_r1": [3.0], "dow": [1.0]}))
    down = o.predict(pd.DataFrame({"spx_r1": [-3.0], "dow": [1.0]}))
    assert up["rally_predicted"] and up["size_multiplier"] == MO.RALLY_SIZE
    assert not down["rally_predicted"] and down["size_multiplier"] == 1.0
    path = o.save(tmp_path / "mo.pkl")
    assert MO.load(path).predict(pd.DataFrame({"spx_r1": [3.0], "dow": [1.0]})) == up
    assert MO.load(path).predict(pd.DataFrame({"spx_r1": [3.0]}))["inputs_missing"] == 1     # tolerated
    assert MO.load(tmp_path / "missing.pkl") is None
