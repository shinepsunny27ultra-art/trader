#!/usr/bin/env python3
"""Gold book jobs (see src/nse_intraday_ai/gold_book.py for the strategy).

    python scripts/gold_book.py preopen [--push]   # 09:00:30 IST: lender trades (pre-open entry)
    python scripts/gold_book.py open [--push]      # 09:16 IST: fill the slots with gap fades
    python scripts/gold_book.py record [--push]    # after the close: paper result

Runs on the laptop (systemd) and on GitHub Actions (.github/workflows/gold_book.yml).
Whichever runs first publishes a stage; a stage already on `main` is not
overwritten by an earlier stage, so the two never fight.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from nse_intraday_ai import gold_book as GB

IST = ZoneInfo("Asia/Kolkata")
STAGES = {"preopen": 1, "open": 2}
LENDER_CUTOFF = "09:08"        # the pre-open order window closes
GAP_FADE_CUTOFF = "09:22"      # the gap fades are booked at the 09:20 bar's open
FORWARD_START = date(2026, 11, 25)
WINDOWS = {"preopen": ("09:00", "09:07"), "open": ("09:15", "09:30")}


def push(title: str, body: str, *, enabled: bool) -> None:
    print(f"\n[{title}]\n{body}")
    if enabled:
        from nse_intraday_ai.alerts import send_ntfy
        if not send_ntfy(body, title=title, priority="high", tags="moneybag"):
            print("  (push failed)")


def in_actions() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS"))


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def remote_stage(read, session: date) -> int:
    """Stage of today's gold list on `main` (0 = none)."""
    raw = read(rel(GB.PICKS))
    try:
        payload = json.loads(raw) if raw else None
    except ValueError:
        return 0
    if not payload or payload.get("session") != session.isoformat():
        return 0
    return STAGES.get(payload.get("stage"), 0)


def publish(files: list[Path], message: str, guard=None) -> bool:
    from nse_intraday_ai import cloud_sync
    return cloud_sync.publish(files, message, force=in_actions(), attempts=3, guard=guard)


def ensure_history(session: date, log=print) -> None:
    """NSE bhavcopies through the previous session (a cloud runner may start behind)."""
    from nse_intraday_ai import nse_bhav
    nse_bhav.backfill(session - timedelta(days=440 if in_actions() else 15), session - timedelta(days=1),
                      workers=2, log=log)


def opens_now(session: date, log=print) -> tuple[dict[str, float], str]:
    """Today's opening prices: the archived/live pre-open auction, else Yahoo's 09:15 bar."""
    from nse_intraday_ai.nse_preopen import archived_opens
    opens = archived_opens(session)
    if opens:
        return opens, "pre-open auction (archived)"
    try:
        from nse_intraday_ai import nse_preopen
        frame = nse_preopen.final_for(session, nse_preopen.fetch(),
                                      now=datetime.now(IST).replace(tzinfo=None))
        eq = frame[(frame["series"] == "EQ") & (frame["iep"] > 0)]
        if not eq.empty:
            return dict(zip(eq["symbol"], eq["iep"].astype(float))), "pre-open auction (NSE)"
    except Exception as exc:                                  # noqa: BLE001 — blocked abroad, or late
        log(f"  pre-open feed unavailable ({type(exc).__name__}); using Yahoo's 09:15 bar")
    from nse_intraday_ai import gap_reversal as G
    syms = [s + ".NS" for s in GB.UNIVERSE]
    G.refresh(syms, interval="5m", period="1d", log=log)
    out = {}
    for s in syms:
        bars = G.load_session_bars(s, session)
        if not bars.empty and 0 in bars.index:
            out[s.removesuffix(".NS")] = float(bars.at[0, "open"])
    return out, "Yahoo 09:15 bar"


def message(payload: dict) -> tuple[str, str]:
    g = payload["gold"]
    head = (f"Gold {g['overnight_pct']:+.2f}% overnight · 20-day {g['trend20_pct']:+.1f}% · "
            f"trend {g['trend']}") if g.get("overnight_pct") is not None else "Gold price unavailable"
    lines = [head]
    for p in payload["picks"]:
        verb = "BUY" if p["side"] == "LONG" else "SELL SHORT"
        lines.append(f"{verb} {p['quantity']} {p['symbol']} ({p['rule']}) · stop {p['stop_pct']:.1f}% · "
                     f"enter {p['entry'].split(' (')[0]}")
    if not payload["picks"]:
        lines.append("No gold trade" + (" yet — gap fades come at 09:16" if payload["stage"] == "preopen" else " today"))
    lines.append("Cover at 15:15. Paper-tracked, weak edge (Sharpe ~1): keep it small.")
    stage = "pre-open" if payload["stage"] == "preopen" else "final"
    return f"🥇 Gold stocks ({stage}) {payload['session']}", "\n".join(lines)


def cmd_stage(args, stage: str) -> None:
    from nse_intraday_ai.nse_calendar import is_trading_day
    now = datetime.now(IST)
    session = date.fromisoformat(args.date) if args.date else now.date()
    if not is_trading_day(session):
        print(f"{session}: not a trading day")
        return
    start, end = WINDOWS[stage]
    if not args.date and not start <= now.strftime("%H:%M") <= end:
        print(f"{now:%H:%M}: outside the {stage} window {start}-{end} — a list now could not be traded as tested")
        return
    from nse_intraday_ai import cloud_sync
    if not args.dry_run and (in_actions() or not args.date):
        def ahead(read) -> bool:
            return remote_stage(read, session) < STAGES[stage]
        if not ahead(lambda r: cloud_sync.read_remote(ROOT / r)):
            print(f"{session}: the {stage} gold list is already on main — standing aside")
            return
    ensure_history(session)
    prior = GB.load()
    if not prior or prior.get("session") != session.isoformat():
        prior = GB.remote_payload(session)                    # the other machine's 09:01 list
    keep = None
    if stage == "open":
        # The lender trades are entered in the pre-open: only those that went out at
        # 09:01 count; with no 09:01 list, the open list is gap fades only.
        same = prior and prior.get("session") == session.isoformat()
        in_time = same and "LENDERS" in tradable_rules(prior)          # a late 09:01 list was not tradable
        keep = [p for p in prior.get("picks", []) if p["rule"] == "LENDERS"] if in_time else []
    opens, source = opens_now(session) if stage == "open" else ({}, None)
    payload = GB.build(session, stage=stage, opens=opens, keep=keep)
    payload["opens_from"] = source
    payload["publisher"] = "github-actions" if in_actions() else "laptop"
    stamp = now.isoformat(timespec="seconds")
    same_day = prior and prior.get("session") == session.isoformat()
    payload["published"] = {**((prior or {}).get("published", {}) if same_day else {}), stage: stamp}
    title, body = message(payload)
    if args.dry_run:
        print(f"DRY RUN — would publish:\n[{title}]\n{body}")
        return
    GB.save(payload)
    files = [GB.PICKS, GB.dated(session), GB.GOLD_MARKS]
    ok = publish(files, f"chore(data): gold list ({stage}) for {session} [skip ci]",
                 guard=lambda read: remote_stage(read, session) < STAGES[stage])
    if in_actions() and not ok:
        print("main got this stage first — not pushing to the phone")
        return
    push(title, body, enabled=args.push)


def tradable_rules(payload: dict) -> set[str]:
    """Rules whose list went out in time to be traded as tested."""
    pub = payload.get("published", {})
    ok = set()
    if pub.get("preopen", "")[11:16] and pub["preopen"][:10] == payload["session"] \
            and pub["preopen"][11:16] <= LENDER_CUTOFF:
        ok.add("LENDERS")
    t = pub.get("open") or ""
    if t[:10] == payload["session"] and t[11:16] <= GAP_FADE_CUTOFF:
        ok.add("GAP_FADE")
    return ok


def record_session(session: date, payload: dict) -> list[dict] | None:
    from nse_intraday_ai import gap_reversal as G
    from nse_intraday_ai.nse_preopen import archived_opens
    picks = [p for p in payload.get("picks", []) if p["rule"] in tradable_rules(payload)]
    if not picks:
        return []
    G.refresh([p["symbol"] + ".NS" for p in picks], interval="5m", period="60d")
    official = archived_opens(session)
    rows = []
    past = session < datetime.now(IST).date()
    for p in picks:
        bars = G.load_session_bars(p["symbol"] + ".NS", session)
        if bars.empty and past:
            print(f"  {session} {p['symbol']}: no 5-minute bars (suspended or not on Yahoo) — no trade")
            continue
        if bars.empty or (G.SQUARE_OFF_BAR not in bars.index and bars.index.max() < G.SQUARE_OFF_BAR - 1):
            return None                                       # not complete yet: retry next run
        t = GB.simulate(p, bars, GB.entry_bar(p["rule"]), G.SQUARE_OFF_BAR,
                        open_price=official.get(p["symbol"]))
        if t is not None:
            rows.append({"session": session.isoformat(), **t})
    return rows


def cmd_record(args) -> None:
    from nse_intraday_ai import cloud_sync
    from nse_intraday_ai.nse_calendar import is_trading_day
    book = pd.read_csv(GB.PAPER) if GB.PAPER.exists() else pd.DataFrame()
    raw = cloud_sync.read_remote(GB.PAPER)
    remote = pd.read_csv(io.StringIO(raw)) if raw else pd.DataFrame()
    if not remote.empty:                                      # adopt what the other machine recorded
        have = set(book["session"]) if not book.empty else set()
        book = pd.concat([book, remote[~remote["session"].isin(have)]], ignore_index=True)
    done = set(book["session"]) if not book.empty else set()
    now = datetime.now(IST)
    last = now.date() if now.strftime("%H:%M") >= "15:30" else now.date() - timedelta(days=1)
    sessions = [d.date() for d in pd.bdate_range(max(FORWARD_START, last - timedelta(days=45)), last)
                if is_trading_day(d.date()) and d.date().isoformat() not in done]
    recorded = []
    for s in sessions:
        payload = GB.load(s) or GB.remote_payload(s)
        if not payload:
            print(f"{s}: no gold list — nothing to record")
            continue
        rows = record_session(s, payload)
        if rows is None:
            print(f"{s}: intraday bars incomplete — retried next run")
            continue
        if not rows:                                          # a no-trade day is still a recorded day
            rows = [{"session": s.isoformat(), "symbol": "", "side": "", "rule": "NONE", "entry": 0, "exit": 0,
                     "quantity": 0, "exit_reason": "NO_TRADE", "exit_time": "", "gross": 0, "costs": 0, "net": 0}]
        book = pd.concat([book, pd.DataFrame(rows)], ignore_index=True)
        recorded.append((s, rows))
    if not recorded:
        print("nothing new to record")
        return
    GB.OUT.mkdir(parents=True, exist_ok=True)
    book.sort_values(["session", "symbol"]).to_csv(GB.PAPER, index=False)
    total = book["net"].sum()
    for s, rows in recorded:
        traded = [r for r in rows if r["rule"] != "NONE"]
        body = "\n".join(f"{r['symbol']:<11} {r['side']:<5} {r['exit_reason']:<10} ₹{r['net']:+,.0f}" for r in traded)
        body = (body or "No gold trade.") + f"\nDay: ₹{sum(r['net'] for r in traded):+,.0f} · gold book ₹{total:+,.0f}"
        push(f"🥇 Gold result {s:%d %b}", body, enabled=args.push and s == recorded[-1][0])
    new = {s.isoformat() for s, _ in recorded}

    def nobody_recorded(read) -> bool:
        r = read(rel(GB.PAPER))
        return not r or not (new & set(pd.read_csv(io.StringIO(r))["session"]))
    publish([GB.PAPER], f"chore(data): gold paper result for {recorded[-1][0]} [skip ci]",
            guard=nobody_recorded if in_actions() else None)


def wait_for_clock() -> None:
    """The laptop's clock can boot ~5.5 h fast until NTP syncs (see gap_reversal.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gap_script", ROOT / "scripts" / "gap_reversal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.wait_for_clock()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preopen", "open", "record"):
        p = sub.add_parser(name)
        p.add_argument("--push", action="store_true", help="send to the phone via ntfy")
        p.add_argument("--date", help="session (YYYY-MM-DD); default today")
        p.add_argument("--dry-run", action="store_true", help="build and print; save and publish nothing")
    a = ap.parse_args()
    if not in_actions():
        wait_for_clock()
    if a.cmd == "record":
        cmd_record(a)
    else:
        cmd_stage(a, a.cmd)


if __name__ == "__main__":
    main()
