"""Gold book: rules, gold marks at the NSE clock, the live list, the simulator, the page."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import gold_book as GB


def _bhav(days: int = 320, seed: int = 0) -> pd.DataFrame:
    """Synthetic bhavcopies of five universe names over `days` sessions."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2025-06-02", periods=days)
    rows = []
    for sym, px in (("MUTHOOTFIN", 2500.0), ("MANAPPURAM", 250.0), ("IIFL", 600.0),
                    ("TITAN", 3500.0), ("KALYANKJIL", 500.0)):
        close = px
        for d in sessions:
            prev = close
            op = prev * (1 + rng.normal(0, 0.01))
            close = op * (1 + rng.normal(0, 0.012))
            hi, lo = max(op, close) * 1.004, min(op, close) * 0.996
            rows.append(dict(session=d.date(), symbol=sym, series="EQ", open=op, high=hi, low=lo, close=close,
                             prev_close=prev, volume=1e6, value=5e8, trades=1e4, deliv_qty=5e5, deliv_pct=50.0))
    return pd.DataFrame(rows)


def _marks(sessions, g_on_last: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(list(sessions)))
    p1530 = pd.Series(2000 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.008, len(idx))), index=idx)
    p0900 = p1530.shift(1) * (1 + np.random.default_rng(2).normal(0, 0.004, len(idx)))
    p0900.iloc[-1] = p1530.iloc[-2] * (1 + g_on_last)
    p1530.iloc[-1] = np.nan                                     # today's close is not known yet
    return pd.DataFrame({"p0900": p0900, "p1530": p1530})


def _live(g_on: float, opens_shift: dict[str, float] | None = None, stage: str = "open", keep=None):
    b = _bhav()
    day = pd.Timestamp(b["session"].max()) + pd.offsets.BDay(1)
    sessions = sorted(pd.to_datetime(b["session"].unique())) + [day]
    last = b[b["session"] == b["session"].max()].set_index("symbol")["close"]
    opens = {s: float(c) * (1 + (opens_shift or {}).get(s, 0.0)) for s, c in last.items()}
    return GB.build(day.date(), stage=stage, opens=opens if stage == "open" else None, bhav=b,
                    gold=_marks(sessions, g_on), keep=keep)


def test_gold_marks_never_look_past_the_clock():
    ts = pd.date_range("2026-11-20 03:00", periods=120, freq="min", tz="UTC")   # 08:30..10:29 IST
    prices = pd.Series(np.arange(120, dtype=float), index=ts)
    m = GB.gold_marks(prices, [date(2026, 11, 20)])
    assert m["p0900"].iloc[0] == 30.0                                         # the 09:00 IST bar, not later
    assert np.isnan(m["p1530"].iloc[0]) or m["p1530"].iloc[0] <= 119          # stale by 15:30: NaN


def test_big_gold_move_trades_the_three_lenders_in_its_direction():
    up = _live(+0.008)
    assert {(p["symbol"], p["side"], p["rule"]) for p in up["picks"]} == \
        {(s, "LONG", "LENDERS") for s in GB.LENDERS}
    down = _live(-0.008)
    assert {p["side"] for p in down["picks"]} == {"SHORT"} and len(down["picks"]) == 3


def test_small_gold_move_no_lenders_gap_fade_fills_at_most_three():
    out = _live(0.001, {"TITAN": 0.04, "KALYANKJIL": -0.05})
    picks = {p["symbol"]: p for p in out["picks"]}
    assert len(out["picks"]) <= GB.K
    assert picks["TITAN"]["side"] == "SHORT" and picks["KALYANKJIL"]["side"] == "LONG"
    assert all(p["rule"] == "GAP_FADE" for p in out["picks"])
    assert out["picks"][0]["symbol"] == "KALYANKJIL"                          # the largest unexplained gap first


def test_preopen_stage_has_no_gap_fades():
    out = _live(0.001, stage="preopen")
    assert out["picks"] == [] and out["stage"] == "preopen"


def test_open_stage_keeps_the_published_lender_trades():
    pre = _live(+0.008, stage="preopen")
    kept = pre["picks"][:2]
    out = _live(0.001, {"TITAN": 0.06}, keep=kept)     # the 09:00 mark moved under the threshold since
    assert [p["symbol"] for p in out["picks"][:2]] == [p["symbol"] for p in kept]
    assert out["picks"][2]["symbol"] == "TITAN" and len(out["picks"]) == 3


def _bars(path: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(path, columns=["open", "high", "low", "close"])


@pytest.mark.parametrize("side,low,high,reason", [("LONG", 97.0, 101, "STOP"), ("SHORT", 99, 103.0, "STOP"),
                                                  ("LONG", 99.5, 101, "SQUARE_OFF")])
def test_simulate_long_and_short_with_stops(side, low, high, reason):
    flat = [(100, 100.5, 99.5, 100)] * 80
    bars = _bars(flat)
    bars.loc[5, ["low", "high"]] = [low, high]
    pick = {"symbol": "X", "side": side, "rule": "LENDERS", "stop_pct": 2.0}
    t = GB.simulate(pick, bars, entry_bar=0, square_off_bar=72)
    assert t["exit_reason"] == reason
    if reason == "STOP":
        assert t["exit"] == pytest.approx(98.0 if side == "LONG" else 102.0)
        assert t["gross"] < 0


def test_gold_page_renders_a_list_and_a_record(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    payload = _live(+0.008)
    picks, paper = tmp_path / "picks.json", tmp_path / "paper.csv"
    picks.write_text(json.dumps(payload))
    pd.DataFrame([{"session": "2026-11-25", "symbol": "MUTHOOTFIN", "side": "LONG", "rule": "LENDERS",
                   "entry": 2500.0, "exit": 2530.0, "quantity": 50, "exit_reason": "SQUARE_OFF",
                   "exit_time": "15:15", "gross": 1500.0, "costs": 160.0, "net": 1340.0}]).to_csv(paper, index=False)
    monkeypatch.setattr(GB, "PICKS", picks)
    monkeypatch.setattr(GB, "PAPER", paper)
    monkeypatch.setenv("NSE_UI_LIVE", "0")
    script = Path(__file__).with_name("_gold_page_app.py")
    script.write_text("from nse_intraday_ai import gold_ui\ngold_ui.page_gold()\n")
    try:
        at = AppTest.from_file(str(script), default_timeout=60).run(timeout=60)
    finally:
        script.unlink()
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(m.value) for m in at.metric)
    assert "+0.80%" in shown


def _gs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gold_script", Path(__file__).resolve().parents[1] / "scripts" / "gold_book.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_lists_published_in_time_are_tradable():
    gs = _gs()
    on_time = {"session": "2026-11-25", "published": {"preopen": "2026-11-25T09:00:41+05:30",
                                                      "open": "2026-11-25T09:16:20+05:30"}}
    assert gs.tradable_rules(on_time) == {"LENDERS", "GAP_FADE"}
    late = {"session": "2026-11-25", "published": {"preopen": "2026-11-25T09:12:00+05:30",
                                                   "open": "2026-11-25T09:40:00+05:30"}}
    assert gs.tradable_rules(late) == set()
    yesterday = {"session": "2026-11-25", "published": {"preopen": "2026-11-24T09:00:00+05:30"}}
    assert gs.tradable_rules(yesterday) == set()
