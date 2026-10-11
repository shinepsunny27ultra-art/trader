#!/usr/bin/env python3
"""Intraday NSE-500 monitor: one scan per run (systemd timer, every minute; signals use completed 5-minute bars).

    python scripts/intraday_monitor.py scan [--push]          # live, during the session
    python scripts/intraday_monitor.py replay --date 2026-10-01   # past session, no fetch/record
    python scripts/intraday_monitor.py board                  # shadow-board statistics

A scan refreshes the NIFTY-500 5-minute bars, writes data/intraday/live.json for
the dashboard, alerts on the gap book's stops, and after 15:20 records the day's
paper shadow-board trades.  Signals are PAPER ONLY — nothing is pushed as a trade.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai import gap_reversal as G  # noqa: E402
from nse_intraday_ai import intraday_monitor as M  # noqa: E402
from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json  # noqa: E402

IST = M.IST
APP = "nse-intraday-monitor"
DEVELOPER = "Shine"


def context(session: date, symbols: list[str], *, fetch_bands: bool) -> M.Context:
    """Daily context, cached once per session (baseline volume profile is the slow part)."""
    path = M.OUT / f"context_{session:%Y%m%d}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["symbols"]) == symbols:
            return M.Context(z["prev_close"], z["atr"], z["cum_base"], z["band"])
    pc, atr = M.daily_context(session, symbols)
    ctx = M.Context(pc, atr, M.volume_baseline(session, symbols), M.bands_for(session, symbols, fetch=fetch_bands))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, symbols=np.array(symbols), prev_close=ctx.prev_close, atr=ctx.atr,
             cum_base=ctx.cum_base, band=ctx.band)
    for old in sorted(M.OUT.glob("context_*.npz"))[:-5]:          # keep a few days
        old.unlink(missing_ok=True)
    return ctx


def run(session: date, nbar: int, *, fetch: bool, push: bool, record: bool, now: datetime) -> dict:
    picks = G.read_picks()
    book_names = ({str(p["symbol"]).removesuffix(".NS") for p in picks.get("picks", [])}
                  if picks and picks.get("session") == session.isoformat() else set())
    symbols = sorted(set(M.universe()) | book_names)
    if fetch:
        got = G.refresh([f"{s}.NS" for s in symbols], interval="5m", period="1d", batch=100, log=print)
        print(f"  5m bars: {len(got)}/{len(symbols)} symbols")
        G.refresh(list(M.INDICES), interval="5m", period="1d", batch=10, log=lambda *_: None)
    ctx = context(session, symbols, fetch_bands=fetch)
    panel = M.load_day(session, symbols, nbar)
    ann = M.todays_announcements(session, set(symbols)) if fetch else pd.DataFrame(columns=["symbol", "ts", "desc", "text"])
    payload = M.build_live(panel, ctx, ann, picks, now, indices=M.index_context(session, nbar))
    M.write_live(payload)
    print(f"  {session} bar {payload['last_bar']}: {len(payload['in_play'])} in play, "
          f"{len(payload['signals'])} paper signals, {len(payload['announcements'])} announcements")
    if push and payload["gap_book"]:
        alert_path = M.OUT / f"alerts_{session:%Y%m%d}.json"
        sent = set(atomic_read_json(alert_path, default=[]) or [])
        new = M.gap_alerts(payload["gap_book"], sent)
        if new:
            from nse_intraday_ai.alerts import send_ntfy
            for key, msg in new:
                if send_ntfy(msg, title="Gap book stop watch", priority="high", tags="warning"):
                    sent.add(key)
                print(f"  alert: {msg}")
            atomic_write_json(alert_path, sorted(sent))
    if record and nbar > M.SQ_BAR:
        n = M.record_day(session, payload["signals"])
        print(f"  shadow book: {n} paper trades recorded for {session}")
    # Push live data to GitHub on completed 5-min bars so Streamlit Cloud stays live
    if record and now.minute % 5 == 0:
        from nse_intraday_ai import cloud_sync

        cloud_sync.publish([M.LIVE_PATH, M.BOOK_PATH], f"chore(data): live sync bar {payload.get('last_bar')} [skip ci]",
                           timeout=25)
    return payload


def missing_board_sessions(today: date) -> list[date]:
    """Trading sessions since the board started (before `today`) not graded yet."""
    from datetime import timedelta

    from nse_intraday_ai.nse_calendar import is_trading_day
    done = set((atomic_read_json(M.GRADED_PATH, default={}) or {}).get("graded", []))
    book = M._read_book(M.BOOK_PATH)
    if not book.empty:
        done |= set(book["session"].astype(str))
    out, day = [], M.BOARD_START
    while day < today:
        if is_trading_day(day) and day.isoformat() not in done:
            out.append(day)
        day += timedelta(days=1)
    return out


def catch_up(today: date, *, fetch: bool = True) -> int:
    """Grade missed sessions (laptop off at 15:15) from cached / re-fetched 5-minute bars.

    Each missed session is attempted at most once per calendar day, so the
    every-minute scan does not re-download a month of bars each time.
    """
    state = atomic_read_json(M.GRADED_PATH, default={}) or {}
    graded, tried = set(state.get("graded", [])), dict(state.get("tried", {}))
    missing = [d for d in missing_board_sessions(today) if tried.get(d.isoformat()) != today.isoformat()]
    if not missing:
        return 0
    if fetch:
        age = (today - missing[0]).days
        period = "5d" if age <= 5 else "1mo" if age <= 28 else "60d"
        got = G.refresh([f"{s}.NS" for s in M.universe()], interval="5m", period=period, batch=100,
                        log=lambda *_: None)
        print(f"  catch-up: refreshed {len(got)} symbols ({period}) for {len(missing)} missed session(s)")
    recorded, symbols = 0, M.universe()
    for session in missing:
        tried[session.isoformat()] = today.isoformat()
        panel = M.load_day(session, symbols, M.NBARS)
        if np.isfinite(panel.c).sum() < 0.5 * len(symbols) * M.NBARS:
            print(f"  catch-up: {session} has too few 5-minute bars, skipped (retry tomorrow)")
            continue
        uni = set(symbols)
        sigs = [s for s in M.run_strategies(panel, context(session, symbols, fetch_bands=False))
                if s["symbol"] in uni]
        n = M.record_day(session, sigs)
        graded.add(session.isoformat())
        recorded += 1
        print(f"  catch-up: {session} — {n} paper trades recorded")
    atomic_write_json(M.GRADED_PATH, {"graded": sorted(graded), "tried": tried})
    return recorded


def cmd_scan(args) -> None:
    now = datetime.now(IST)
    from nse_intraday_ai.nse_calendar import is_trading_day
    if not is_trading_day(now.date()):
        print(f"{now.date()}: not a trading day")
        return
    hm = now.hour * 60 + now.minute
    if not args.force and not (9 * 60 + 20 <= hm <= 15 * 60 + 45):
        print(f"{now:%H:%M}: outside the scan window (09:20-15:45)")
        return
    try:
        catch_up(now.date())
    except Exception as exc:                             # noqa: BLE001 — never block the live scan
        print(f"  catch-up failed: {exc}")
    run(now.date(), M.completed_bars(now), fetch=True, push=args.push, record=True, now=now)


def cmd_catchup(_args) -> None:
    n = catch_up(datetime.now(IST).date())
    print(f"  {n} session(s) caught up")


def cmd_replay(args) -> None:
    session = date.fromisoformat(args.date)
    payload = run(session, args.bars, fetch=False, push=False, record=False,
                  now=datetime.combine(session, datetime.min.time(), IST).replace(hour=15, minute=30))
    sig = pd.DataFrame(payload["signals"])
    if not sig.empty:
        print(sig[["strategy", "symbol", "side", "time", "entry", "stop", "target", "status", "exit", "net_bps"]]
              .to_string(index=False))
        print(sig.groupby("strategy").net_bps.agg(["count", "mean"]).round(1).to_string())


def cmd_board(_args) -> None:
    book = M._read_book(M.BOOK_PATH)
    print(M.board_stats(book).to_string(index=False))


def main() -> None:
    print(f"{APP} | build {datetime.fromtimestamp(Path(__file__).stat().st_mtime):%Y-%m-%d} | developer {DEVELOPER}")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--push", action="store_true", help="phone alerts for the gap book's stops")
    s.add_argument("--force", action="store_true", help="scan outside 09:20-15:45")
    s.set_defaults(fn=cmd_scan)
    r = sub.add_parser("replay")
    r.add_argument("--date", required=True)
    r.add_argument("--bars", type=int, default=M.NBARS)
    r.set_defaults(fn=cmd_replay)
    sub.add_parser("board").set_defaults(fn=cmd_board)
    sub.add_parser("catchup", help="grade shadow-board sessions missed while the machine was off"
                   ).set_defaults(fn=cmd_catchup)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
