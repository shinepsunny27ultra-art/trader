"""Market outlook: is today likely to be a broad rally from the open?  A size signal.

The short book's day moves against the market's open->close move (corr ~ -0.6 on the
walk-forward book), and it loses on broad intraday rallies.  A small gradient-boosted
model on the market-level inputs known at 08:15 (global closes, NIFTY/BANK/midcap,
India VIX, participant positioning, breadth, calendar) predicts the day's median
open->close move of the liquid universe.  It is weak but consistent out of sample
(research/gap_book_2026_10/market_direction.py, yearly walk-forward 2019..2026-10):
direction right ~60% of days, and the book by predicted tercile was +82/+62/+51 bps
(2019..2025-08) and +66/+55/+23 (2025-09..2026-10, down/middle/rally).  Halving the
size on predicted-rally days (cut at the 2/3 quantile of the model's own out-of-sample
predictions) raised net per unit of capital by ~4 bps and the Sharpe 5.29 -> 5.49
in the held-out year.  It never changes which stocks are shorted.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "data" / "models" / "market_outlook.pkl"
RALLY_SIZE = 0.5                      # size multiplier on a predicted-rally day
UNIVERSE = 300


def _params() -> dict:
    return dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=60,
                l2_regularization=5.0, random_state=0)


def market_columns(panel: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Numeric columns with one value per session (checked on ~150 sampled sessions).

    Today's opening-gap family (gap0, x_gap0_*) is not known at 08:15 and is left out.
    """
    from nse_intraday_ai.ranker import NON_FEATURES

    sess = panel.index.get_level_values("session")
    days = pd.Index(sorted(set(sess)))
    sample = panel[sess.isin(days[:: max(1, len(days) // 150)])]
    num = [c for c in panel.columns if panel[c].dtype.kind == "f" and c not in NON_FEATURES
           and c not in exclude and "gap0" not in c]
    nuniq = sample[num].groupby(sample.index.get_level_values("session")).nunique().max()
    filled = sample[num].notna().mean()
    return [c for c in num if nuniq[c] <= 1 and filled[c] > 0.05]       # never-filled columns teach nothing


def daily_frame(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per session: the market inputs and (if labelled) the day's median open->close."""
    u = panel[panel["turn_rank"] <= UNIVERSE]
    sess = u.index.get_level_values("session")
    df = u[cols].groupby(sess).first()
    if "intra_bps" in u:
        df["target"] = u["intra_bps"].groupby(sess).median()
    return df


@dataclass
class Outlook:
    features: list[str]
    model: object = field(repr=False)
    rally_cut_bps: float                 # predictions above this = "rally predicted"
    trained_through: str
    oos: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, panel: pd.DataFrame, exclude: tuple[str, ...] = (), eval_years: int = 2) -> "Outlook":
        from sklearn.ensemble import HistGradientBoostingRegressor

        cols = market_columns(panel, exclude)
        df = daily_frame(panel, cols).dropna(subset=["target"])
        df.index = pd.to_datetime(df.index)
        # The rally cut comes from the model's own out-of-sample predictions over the
        # last `eval_years` (yearly refits), never from in-sample fits.
        last = df.index.max()
        oos = pd.Series(np.nan, index=df.index)
        for year in range(last.year - eval_years, last.year + 1):
            train = df[df.index < f"{year}-01-01"]
            test = df[(df.index >= f"{year}-01-01") & (df.index < f"{year + 1}-01-01")]
            if len(train) < 400 or test.empty:
                continue
            m = HistGradientBoostingRegressor(**_params()).fit(train[cols], train["target"].clip(-300, 300))
            oos[test.index] = m.predict(test[cols])
        oos = oos.dropna()
        cut = float(oos.quantile(2 / 3)) if len(oos) >= 100 else 0.0
        model = HistGradientBoostingRegressor(**_params()).fit(df[cols], df["target"].clip(-300, 300))
        hit = float((np.sign(oos) == np.sign(df.loc[oos.index, "target"])).mean()) if len(oos) else float("nan")
        return cls(features=cols, model=model, rally_cut_bps=round(cut, 2), trained_through=str(last.date()),
                   oos={"sessions": int(len(oos)), "direction_hit": round(hit, 3),
                        "corr": round(float(oos.corr(df.loc[oos.index, "target"])), 3) if len(oos) else None})

    def predict(self, rows: pd.DataFrame) -> dict:
        """The outlook for the session in `rows` (market columns are the same on every row)."""
        X = rows.reindex(columns=self.features).iloc[[0]].astype(float)
        missing = [c for c in self.features if c not in rows.columns or X[c].isna().all()]
        pred = float(self.model.predict(X)[0])
        rally = pred > self.rally_cut_bps
        return {"predicted_open_to_close_bps": round(pred, 1), "rally_cut_bps": self.rally_cut_bps,
                "rally_predicted": bool(rally), "size_multiplier": RALLY_SIZE if rally else 1.0,
                "trained_through": self.trained_through, "inputs_missing": len(missing)}

    def save(self, path: Path = PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({**self.__dict__, "saved_at": datetime.now().isoformat(timespec="seconds")},
                                     protocol=5))
        tmp.replace(path)
        return path


def load(path: Path = PATH) -> Outlook | None:
    try:
        raw = pickle.loads(path.read_bytes())
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    raw.pop("saved_at", None)
    return Outlook(**raw)
