"""🥇 Gold stocks: a separate page for the gold book (src/nse_intraday_ai/gold_book.py).

Reads what scripts/gold_book.py writes: data/gold_book/picks.json (today's <= 3
names), gold_marks.csv (gold at 09:00 / 15:30 IST each session) and
paper_book.csv (the forward paper record).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nse_intraday_ai import gold_book as GB
from nse_intraday_ai.gap_reversal_ui import (
    IST,
    PALETTE,
    _signed,
    chart,
    expected_session,
    read_csv,
    read_json,
    table,
)

GOLD = "#FFC94A"


def _gold_marks() -> pd.DataFrame | None:
    m = read_csv(str(GB.GOLD_MARKS))
    if m is None or m.empty:
        return None
    m["session"] = pd.to_datetime(m["session"])
    return m.dropna(subset=["p1530"]).set_index("session")


def _trend_panel(payload: dict | None) -> None:
    g = (payload or {}).get("gold") or {}
    trend = g.get("trend", "unknown")
    c = st.columns(4)
    on = g.get("overnight_pct")
    c[0].metric("Gold overnight", "—" if on is None else f"{on:+.2f}%",
                None if on is None else ("lenders trade today" if abs(on) >= GB.GOLD_MOVE * 100
                                         else f"under ±{GB.GOLD_MOVE * 100:.1f}%: no lender trade"),
                delta_color="off")
    t20 = g.get("trend20_pct")
    c[1].metric("Gold, 20 sessions", "—" if t20 is None else f"{t20:+.1f}%")
    ma = g.get("vs_ma50_pct")
    c[2].metric("vs 50-day average", "—" if ma is None else f"{ma:+.1f}%")
    c[3].metric("Gold trend", {"UP": "▲ UP", "DOWN": "▼ DOWN", "MIXED": "◆ MIXED"}.get(trend, "—"))

    m = _gold_marks()
    if m is not None:
        m = m.tail(250)
        ma50 = m["p1530"].rolling(50, min_periods=40).mean()
        fig = go.Figure()
        fig.add_scatter(x=m.index, y=m["p1530"], name="Gold at 15:30 IST (US$/oz)", line=dict(color=GOLD, width=2.2))
        fig.add_scatter(x=m.index, y=ma50, name="50-session average", line=dict(color=PALETTE[1], dash="dash"))
        fig.update_layout(title="Spot gold at the NSE close")
        fig.update_yaxes(tickprefix="$")
        chart(fig, m.assign(ma50=ma50).reset_index(), "gold_trend", height=320)


def _today(payload: dict | None, now: datetime) -> None:
    st.subheader("Today's gold trades (at most 3)")
    if not payload:
        st.info("No gold list yet. It is published at 09:01 IST (lender trades, pre-open) and completed "
                "at 09:16 IST (gap fades) by `scripts/gold_book.py` — on the laptop or on GitHub.")
        return
    session = payload.get("session", "?")
    expected = expected_session(now).isoformat()
    if session != expected and session != now.date().isoformat():
        st.warning(f"This list is for **{session}** — today's is published from 09:01 IST. Do not trade it.")
    stage = payload.get("stage")
    pub = payload.get("published", {})
    st.caption(f"Session {session} · {'final (09:16)' if stage == 'open' else 'pre-open (09:01) — gap fades follow at 09:16'}"
               f" · published {', '.join(f'{k} {v[11:16]}' for k, v in pub.items())}"
               f" · by {payload.get('publisher', '?')}"
               + (f" · opens from {payload['opens_from']}" if payload.get("opens_from") else ""))
    picks = payload.get("picks", [])
    if not picks:
        st.success("**No gold trade today.**" + (" Gap fades are checked at 09:16." if stage == "preopen" else
                                                  " Gold moved too little overnight and no gold stock opened far "
                                                  "from what gold explains."))
        return
    rows = []
    for p in picks:
        rows.append({"Action": "BUY" if p["side"] == "LONG" else "SELL SHORT", "Stock": p["symbol"],
                     "Qty": p["quantity"], "Rule": "Lenders follow gold" if p["rule"] == "LENDERS" else "Gap fade",
                     "When": p["entry"], "Stop": f"{p['stop_pct']:.1f}% {'below' if p['side'] == 'LONG' else 'above'} fill",
                     "Why": p["reason"]})
    table(pd.DataFrame(rows), "gold_today")
    st.caption("MIS orders; stop-loss SL-M at the fill ± the stop shown; cover everything at 15:15. "
               f"₹{GB.CAPITAL_PER_NAME / 1e5:.2f} lakh a name.")


def _universe(payload: dict | None) -> None:
    uni = (payload or {}).get("universe")
    if not uni:
        return
    with st.expander("Gold universe — sensitivity to gold and today's gaps"):
        df = pd.DataFrame(uni).rename(columns={"symbol": "Stock", "lender": "Gold lender", "tradable": "Liquid enough",
                                               "prev_close": "Prev close", "gap_pct": "Open gap %",
                                               "beta": "Gap β to gold", "turnover_cr": "Turnover ₹cr/day"})
        table(df.sort_values(["Liquid enough", "Turnover ₹cr/day"], ascending=False), "gold_universe")
        st.caption("β = how much of gold's overnight move a stock's opening gap carries (250 sessions). "
                   "Liquid enough = ≥ ₹10 cr/day traded, price ≥ ₹50, no trade-for-trade days.")


def _record() -> None:
    st.subheader("Paper record")
    book = read_csv(str(GB.PAPER))
    if book is None or book.empty:
        st.info("No paper trades yet — the record starts 2026-11-25 and is written after each close.")
        return
    trades = book[book["rule"] != "NONE"].copy()
    days = book.groupby("session")["net"].sum().reset_index()
    days["cumulative"] = days["net"].cumsum()
    c = st.columns(4)
    c[0].metric("Net P&L", f"₹{book['net'].sum():+,.0f}")
    c[1].metric("Sessions / trades", f"{len(days)} / {len(trades)}")
    c[2].metric("Win rate", f"{(trades['net'] > 0).mean() * 100:.0f}%" if len(trades) else "—")
    if len(trades):
        bps = trades["net"] / (trades["entry"] * trades["quantity"]) * 1e4
        c[3].metric("Net per trade", f"{bps.mean():+.1f} bps", f"backtest {GB.EVIDENCE['hold']['net_bps_per_trade']:+.1f}",
                    delta_color="off")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])
    fig.add_scatter(x=days["session"], y=days["cumulative"], name="Cumulative net", mode="lines+markers",
                    line=dict(width=2.5, color=GOLD), row=1, col=1)
    fig.add_bar(x=days["session"], y=days["net"], name="Daily net", marker_color=_signed(days["net"]), row=2, col=1)
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickprefix="₹")
    chart(fig, days, "gold_equity", height=360)
    if len(trades):
        table(trades.sort_values(["session", "symbol"], ascending=[False, True]), "gold_trades")


def _evidence() -> None:
    e = GB.EVIDENCE
    with st.expander("How this was tested — and how weak it is"):
        st.markdown(
            f"""
**Rules** (at most {GB.K} stocks a day, intraday, stop {GB.STOP_ATR}× ATR, exit 15:15):
1. **Lenders follow gold** — if spot gold moved ≥ {GB.GOLD_MOVE * 100:.1f}% between 15:30 yesterday and 09:00
   today, trade MUTHOOTFIN / MANAPPURAM / IIFL in gold's direction (gold is their loan collateral; their
   intraday move follows the overnight gold move, jewellers' does not).
2. **Gap fade** — slots left go to the gold stocks whose opening gap is ≥ {GB.RESID_MIN * 100:.0f}% away from
   what gold's overnight move explains (gap − β × gold move), faded: long a gap that is too low, short one too high.

**Backtest, 13 bps costs** (+5 for entering after the open):

| | trades | net / trade | t |
|---|---|---|---|
| Development {e['dev']['period']} | {e['dev']['trades']:,} | {e['dev']['net_bps_per_trade']:+.1f} bps | {e['dev']['t']} |
| Hold-out {e['hold']['period']} (looked at once) | {e['hold']['trades']:,} | {e['hold']['net_bps_per_trade']:+.1f} bps | {e['hold']['t']} |

Lender rule {e['lenders']['dev']:+.1f} / {e['lenders']['hold']:+.1f} bps, gap fade {e['gap_fade']['dev']:+.1f} /
{e['gap_fade']['hold']:+.1f} (dev / hold-out). Rejected: following or fading gold with the jewellers, trading the
gold trend directly (−31 bps in the hold-out), the main gap rule on gold names.

**Weak edge:** Sharpe ≈ {e['sharpe']}, worst drawdown ≈ {e['max_drawdown_pct']:.0f}% of the gold book's capital
over 2016-26 (the gap book: Sharpe ≈ 5). Neither rule reached the usual t ≥ 3 bar on development data.
Gap fades are entered ~09:20, later than the backtest's open — the paper record shows what that costs.
Size it small.
""")


def page_gold() -> None:
    now = datetime.now(IST)
    payload = read_json(str(GB.PICKS))
    _trend_panel(payload)
    st.divider()
    _today(payload, now)
    _universe(payload)
    st.divider()
    _record()
    _evidence()
