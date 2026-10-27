"""Run the gap-reversal book: pre-open picks, opening stops, paper record, backtest.

    python scripts/gap_reversal.py picks            # 08:15: the session's short list
    python scripts/gap_reversal.py final            # 09:09: re-rank on today's opening prices
    python scripts/gap_reversal.py levels           # after 09:15: exact stop prices
    python scripts/gap_reversal.py record           # after the close: paper result
    python scripts/gap_reversal.py backtest         # last 14 sessions + dev + decade

`--push` sends the result to the phone (ntfy).  Scheduled by the
nse-gap-{picks,levels,record} systemd timers (see deploy/systemd/).  Strategy
and evidence: src/nse_intraday_ai/gap_reversal.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai import gap_reversal as G  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
OUT = G.OUT_DIR
PAPER = OUT / "paper_book.csv"
BACKTEST = OUT / "backtest.json"
CONFIG = G.GapReversalConfig()
HISTORY_DAYS_DEFAULT = 20          # calendar days of NSE files the morning refresh looks back
# The first session of the forward (never-backtested) paper record.
FORWARD_START = date(2026, 9, 29)


def wait_for_clock(timeout: int = 300) -> None:
    """Block until NTP has synchronised the clock, up to `timeout` seconds.

    On 2026-09-29 the hardware clock came up ~5.5 h fast.  Until NTP fixed it,
    TLS to Yahoo failed ("possibly delisted" for every symbol) and "today"
    meant the wrong session.  Timers fire right after boot, so they must wait.
    """
    import subprocess
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return                       # no timedatectl: nothing to wait for
        if out != "no":
            break
        time.sleep(10)
    else:
        print("warning: clock still not NTP-synchronised; continuing")
    wait_for_network(max(30, int(deadline - time.time())))


def wait_for_network(timeout: int = 180, hosts: tuple[str, ...] = ("query1.finance.yahoo.com",
                                                                   "archives.nseindia.com")) -> bool:
    """Block until DNS resolves, up to `timeout` seconds.

    2026-10-27 19:15: the laptop resumed from sleep, every missed timer fired at
    once and the Wi-Fi was not back yet — all 12 Yahoo downloads failed, `record`
    booked the day on cached bars that stopped at 14:15, and the phone pushes
    failed ("Temporary failure in name resolution").
    """
    import socket
    import time

    deadline = time.time() + timeout
    while True:
        try:
            for host in hosts:
                socket.getaddrinfo(host, 443)
            return True
        except OSError:
            if time.time() >= deadline:
                print("warning: network still unreachable; continuing")
                return False
            time.sleep(5)


def push(title: str, body: str, *, enabled: bool, priority: str = "high") -> None:
    print(f"\n[{title}]\n{body}")
    if not enabled:
        return
    from nse_intraday_ai.alerts import send_ntfy

    if not send_ntfy(body, title=title, priority=priority, tags="chart_with_downwards_trend"):
        print("  (push failed)")


# ── commands ────────────────────────────────────────────────────────────────

def _band_filter(session: date, picks: list[G.Pick], info: dict, *, fetch: bool) -> list[G.Pick]:
    """Drop shorts that can't be placed/protected today (price band, series); never blocks a list."""
    from nse_intraday_ai import nse_bands

    try:
        table = nse_bands.load(session, fetch=fetch)
    except Exception as exc:                                  # noqa: BLE001 — optional input
        print(f"  price bands unavailable ({type(exc).__name__}); list not band-filtered")
        return picks
    if table.empty:
        print("  price bands unavailable; list not band-filtered")
    picks, dropped = G.drop_untradeable(picks, table, CONFIG)
    if dropped:
        print(f"  dropped: {'; '.join(dropped)}")
        info["band_excluded"] = dropped
    return picks


def _publish(session: date | None, *, fetch: bool,
             history_days: int = 20) -> tuple[date, date, list[G.Pick], dict]:
    """The NSE-data pipeline (experts + Hedge blend); the Yahoo gap rule if it fails."""
    from nse_intraday_ai import pipeline as P

    target = session or G.next_session()
    try:
        prev, picks, info = P.morning(target, CONFIG, fetch=fetch, history_days=history_days)
        picks = _band_filter(target, picks, info, fetch=fetch)
        G.with_entry_limits(picks, CONFIG)
        G.save_picks(target, picks, CONFIG, based_on=prev.isoformat(), source="nse", **info)
        return target, prev, picks, info
    except Exception as exc:                                  # noqa: BLE001
        # A broken morning must still produce a correct list if one can be
        # made: the gap rule on Yahoo data has its own freshness guard.
        print(f"NSE pipeline failed ({type(exc).__name__}: {exc}); falling back to the gap rule")
        target, last, picks = G.publish_picks(target, CONFIG, fetch=fetch)
        info = {"fallback": f"{type(exc).__name__}: {exc}"[:200]}
        picks = _band_filter(target, picks, info, fetch=fetch)
        G.with_entry_limits(picks, CONFIG)
        payload = G.read_picks() or {}
        G.save_picks(target, picks, CONFIG, based_on=payload.get("based_on"), source="yahoo-rule", **info)
        return target, last, picks, info


def published_today(payload: dict | None, today: date) -> str | None:
    """Who built `payload` as today's list this morning: "laptop", "github-actions" or None.

    An evening PRELIMINARY list (built the day before) does not count: on a
    morning the laptop is off, the cloud's fresh list replaces it.
    """
    if not payload or payload.get("session") != today.isoformat():
        return None
    if not str(payload.get("generated_at", "")).startswith(today.isoformat()):
        return None
    if payload.get("publisher") in ("laptop", "github-actions"):
        return payload["publisher"]
    # Lists written before the publisher field: only the laptop has a model.
    return "laptop" if payload.get("model") else "github-actions"


def laptop_already_published(payload: dict | None, today: date) -> bool:
    """True if `payload` is the laptop's list for today, built this morning."""
    return published_today(payload, today) == "laptop"


def cmd_picks(args) -> None:
    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else None
    in_actions = bool(os.environ.get("GITHUB_ACTIONS"))
    if session is None:
        from nse_intraday_ai.nse_calendar import is_trading_day
        today = datetime.now(IST).date()
        if today.weekday() < 5 and not is_trading_day(today):
            print(f"{today}: NSE holiday — no list published")
            return
        # The cloud runner has no model or archive: it only stands in for a laptop
        # that has not published today's list (the checkout holds main's copy).
        if in_actions and laptop_already_published(G.read_picks(), today):
            print(f"{today}: the laptop already published today's list — the cloud fallback stands aside")
            return
    history_days = args.history_days
    if in_actions and history_days == HISTORY_DAYS_DEFAULT:
        # A fresh runner starts with an empty data/: the 20-session liquidity rank
        # and the 14-session ATR need a full window (2026-10-06: 12 sessions let
        # FEDDERSHOL and SEDEMAC into the "top 300").
        history_days = 45
    try:
        session, last, picks, info = _publish(session, fetch=not args.no_fetch,
                                              history_days=history_days)
    except G.StaleDataError as exc:
        push("⚠ Gap-reversal: NO LIST", f"{exc}.\nNo picks published — do not trade a list "
             "from an earlier message. Retry: python scripts/gap_reversal.py picks",
             enabled=args.push)
        raise SystemExit(2) from exc

    now = datetime.now(IST)
    title, body, priority = _list_message(session, last, picks, info, now)
    if args.push and not in_actions and session == now.date():
        body = _replaces_fallback_note(session) + body
    push(title, body, enabled=args.push, priority=priority)
    # When the list first went out, and whether that was after the entry cutoff:
    # `record` books only lists someone could have entered at the open.
    G.mark_published(session, now, late=session == now.date() and now.strftime("%H:%M") > G.ENTRY_CUTOFF)
    _sync(f"chore(data): publish picks for {session} [skip ci]")


def _list_message(session: date, last, picks: list[G.Pick], info: dict,
                  now: datetime) -> tuple[str, str, str]:
    """(title, body, priority) of the phone message for a list."""
    main = [p for p in picks if not p.reserve]
    lines = [f"{p.rank}. SHORT {p.quantity} {p.symbol.removesuffix('.NS'):<11} "
             + (f"LIMIT ₹{p.limit_price:,.2f}  " if p.limit_price else "")
             + f"gap yday {p.gap_prev_pct:+.1f}%  stop +₹{p.stop_distance:,.1f} ({p.stop_pct:.1f}%)"
             for p in main]
    reserves = ", ".join(p.symbol.removesuffix(".NS") for p in picks if p.reserve)
    late = session == now.date() and now.strftime("%H:%M") > G.ENTRY_CUTOFF   # auction entry can close from 09:08
    if late:
        lines.insert(0, "⚠ LATE: the open has passed. The tested entry is the open itself — "
                        "entering now is a different, untested trade. Skip today.")
    # A list built before its session's day (2026-10-06 20:31: the boot-time
    # catch-up after a day off) lacks the overnight global closes the model is
    # trained on; the 08:15 run rebuilds it.
    preliminary = session > now.date()
    if preliminary:
        lines.insert(0, f"PRELIMINARY (built {now:%a %H:%M}, before the overnight global closes). "
                        "The 08:15 list replaces it — trade that one.")
    if info.get("drift_alarm") or info.get("cautious_mode"):
        mult = info.get("size_multiplier", 0.6)
        lines.insert(0, f"⚠ DRIFT ALARM / CAUTIOUS MODE: the live book has been running persistently "
                        f"below its backtest. Trade {mult:.0%} of the quantities shown until it clears.")
    if info.get("ranked_by") == "ranker":
        t = info.get("guard_t")
        ranked = ("\nRanked by the trained model"
                  + (f" (60-session edge over the gap rule: t={t:+.1f})." if t is not None else "."))
    elif os.environ.get("GITHUB_ACTIONS"):
        ranked = ("\nCLOUD FALLBACK: the laptop has not published today's list, so this one is ranked by "
                  "the gap rule (no model in the cloud).")
    elif info.get("ranked_by") == "rule" and "fallback" not in info:
        ranked = ("\nRanked by the gap rule — the model's guard has switched it off "
                  f"(t={info.get('guard_t')}) or no model is promoted yet.")
    else:
        ranked = f"\nRanked by the gap rule ({info.get('fallback', 'fallback')})."
    if info.get("band_excluded"):
        ranked += f"\nSkipped (can't be shorted/protected today): {'; '.join(info['band_excluded'])}."
    if info.get("missing_inputs"):
        blank = info["missing_inputs"]
        ranked += (f"\n⚠ {len(blank)} model inputs are blank today ({', '.join(blank[:4])}"
                   f"{', …' if len(blank) > 4 else ''}): a data source is missing, so the ranking is degraded.")
    body = "\n".join(lines) + (
        f"\nIf a name can't be shorted (ASM/T2T), use the next: {reserves}"
        + ("\nEnter in pre-open 09:00-09:08: MIS SELL LIMIT at the price shown. It fills at the "
           "official open only if the stock opens at/above it. At 09:15 CANCEL any unfilled order "
           "— do not chase (names that open lower lose on average)."
           if any(p.limit_price for p in main) else
           "\nEnter: MIS market SELL in pre-open 09:00-09:05 (NSE rejects pre-open market orders "
           "after 09:05), or at 09:15 sharp.")
        + f"\nThen: BUY SL-M at your fill + the stop shown. Cover all at {CONFIG.square_off}."
        f"\n~₹{CONFIG.capital / CONFIG.picks:,.0f} each, based on {last} closes." + ranked
    )
    title = f"{'🕗 Preliminary' if preliminary else '📉'} gap-reversal shorts for {session:%a %d %b}"
    if os.environ.get("GITHUB_ACTIONS"):
        title = f"☁ Cloud-fallback gap-reversal shorts for {session:%a %d %b}"
    return title, body, ("default" if preliminary else "high")


def _remote_picks() -> dict | None:
    """The list currently on GitHub `main` (what the cloud app and the fallback see)."""
    import json

    from nse_intraday_ai import cloud_sync

    raw = cloud_sync.read_remote(G.PICKS_PATH)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _replaces_fallback_note(session: date) -> str:
    """A first line for the laptop's message when the cloud fallback already sent a list."""
    prior = _remote_picks()
    if prior and prior.get("session") == session.isoformat() and \
            published_today(prior, session) == "github-actions":
        return (f"REPLACES the cloud-fallback list sent at {str(prior.get('generated_at', ''))[11:16]} — "
                "trade THIS list, not that one.\n")
    return ""


def _sync(message: str) -> None:
    """Mirror the list and the dashboard state onto GitHub `main` for Streamlit Cloud."""
    from nse_intraday_ai import cloud_sync

    cloud_sync.publish(cloud_sync.state_files(), message)


# All times IST.  The laptop's 08:15 job lands on `main` by ~08:16 (25-45 s run).
FALLBACK_BUILD_AT = "08:00"     # build the gap-rule list from here (it needs no morning data)
FALLBACK_DEADLINE = "08:20"     # the laptop's list has priority until here; publish right after
FALLBACK_POLL_SECONDS = 20


def _at(day: date, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=IST)


def cmd_cloud_fallback(args, *, clock=None, sleep=None, remote=None, publish=None) -> str:
    """GitHub Actions: publish a gap-rule list by ~08:21 IST unless the laptop does.

    1. Sleep until FALLBACK_BUILD_AT (the 03:00 run waits here, so GitHub's late
       starts — 15 min is normal, hours are possible — cannot make it miss 08:30).
    2. Stand aside if today's list is already on `main` (laptop or an earlier run).
    3. Build the list (no push yet), then poll `main` until FALLBACK_DEADLINE:
       if the laptop publishes, stand aside.
    4. Commit it to `main`, then push it to the phone.
    Returns what happened (for the tests and the log).
    """
    import time as _time

    clock = clock or (lambda: datetime.now(IST))
    sleep = sleep or _time.sleep
    remote = remote or _remote_picks
    if publish is None:
        import json

        from nse_intraday_ai import cloud_sync
        rel = G.PICKS_PATH.resolve().relative_to(ROOT.resolve()).as_posix()

        def nobody_published(read) -> bool:
            raw = read(rel)
            try:
                return published_today(json.loads(raw) if raw else None, today) is None
            except ValueError:
                return True

        def publish(msg: str) -> bool:
            # Push both picks.json AND the dated history copy so the laptop's
            # load_picks() can find cloud-fallback picks via read_remote().
            dated = G.dated_picks_path(target)
            files = [G.PICKS_PATH]
            if dated.exists():
                files.append(dated)
            return cloud_sync.publish(files, msg, force=True, attempts=3, guard=nobody_published)

    today = clock().date()
    if not args.any_day:
        from nse_intraday_ai.nse_calendar import is_trading_day
        if not is_trading_day(today):
            print(f"{today}: not an NSE trading day — nothing to do")
            return "holiday"
    build_at, deadline = _at(today, args.build_at), _at(today, args.deadline)
    if clock() < build_at:
        wait = (build_at - clock()).total_seconds()
        print(f"{clock():%H:%M:%S} IST: waiting {wait / 60:.0f} min until {args.build_at} IST", flush=True)
        sleep(wait)
    who = published_today(remote(), today)
    if who:
        print(f"{clock():%H:%M:%S} IST: today's list is already on main ({who}) — standing aside")
        return f"stood aside: {who}"
    target = G.next_session() if args.any_day else today
    try:
        session, last, picks, info = _publish(target, fetch=not args.no_fetch, history_days=args.history_days)
    except G.StaleDataError as exc:
        push("⚠ Gap-reversal: NO LIST (cloud fallback)", f"{exc}.\nNo picks published.", enabled=args.push)
        raise SystemExit(2) from exc
    print(f"{clock():%H:%M:%S} IST: gap-rule list built "
          f"({', '.join(p.symbol.removesuffix('.NS') for p in picks if not p.reserve)})", flush=True)
    while clock() < deadline:
        who = published_today(remote(), today)
        if who:
            print(f"{clock():%H:%M:%S} IST: the {who} list arrived — standing aside")
            return f"stood aside: {who}"
        sleep(max(1.0, min(FALLBACK_POLL_SECONDS, (deadline - clock()).total_seconds())))
    who = published_today(remote(), today)
    if who:
        print(f"{clock():%H:%M:%S} IST: today's list is already on main ({who}) — standing aside")
        return f"stood aside: {who}"
    title, body, priority = _list_message(session, last, picks, info, clock())
    if args.dry_run:
        print(f"{clock():%H:%M:%S} IST: DRY RUN — would publish and push:\n[{title}]\n{body}")
        return "dry run"
    if not publish(f"chore(data): cloud-fallback picks for {session} [skip ci]"):
        who = published_today(remote(), today)
        if who:
            print(f"{clock():%H:%M:%S} IST: the {who} list landed first — standing aside")
            return f"stood aside: {who}"
        raise SystemExit("could not publish the fallback list to main")
    print(f"{clock():%H:%M:%S} IST: published to main")
    push(title, body, enabled=args.push, priority=priority)
    return "published"


def cmd_final(args) -> None:
    """09:09 — re-rank with the auction's opening prices and push the FINAL list."""
    from nse_intraday_ai import pipeline as P

    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    payload = G.read_picks()
    if payload is None or payload.get("session") != session.isoformat():
        print(f"no 08:15 list for {session}; nothing to re-rank")
        return
    try:
        result = P.open_rerank(session, CONFIG)
    except Exception as exc:                                  # noqa: BLE001
        print(f"09:09 re-rank failed ({type(exc).__name__}: {exc}); the 08:15 list stands")
        push(f"Gap-reversal {session:%d %b}: 08:15 list stands",
             f"The 09:09 re-rank could not run ({type(exc).__name__}). Trade the 08:15 list.",
             enabled=args.push, priority="default")
        return
    if result is None:
        # No open model promoted, or its guard is off: the 08:15 list already is
        # the final one, and a second message saying so would only be noise.
        print("the 08:15 list stands (no promoted open model, or its guard is off)")
        return
    picks, info = result
    picks = _band_filter(session, picks, info, fetch=True)
    preliminary = [p["symbol"] for p in payload.get("picks", []) if not p.get("reserve")]
    extra = {k: v for k, v in payload.items() if k not in ("session", "generated_at", "config", "picks")}
    G.save_picks(session, picks, CONFIG, **{**extra, **info, "preliminary": preliminary,
                                            "final_at": datetime.now(IST).isoformat(timespec="seconds")})
    main = [p for p in picks if not p.reserve]
    reserves = ", ".join(p.symbol.removesuffix(".NS") for p in picks if p.reserve)
    changed = len({p.symbol for p in main} - set(preliminary))
    lines = [f"{p.rank}. SHORT {p.quantity} {p.symbol.removesuffix('.NS'):<11} "
             f"stop +₹{p.stop_distance:,.1f} ({p.stop_pct:.1f}%)"
             for p in main]
    body = "\n".join(lines) + (
        f"\nReserves: {reserves}"
        f"\nEnter: MIS market SELL at 09:15:00 sharp (this list uses today's opening prices)."
        f"\nThen: BUY SL-M at your fill + the stop shown. Cover all at {CONFIG.square_off}."
        f"\n{changed} of 8 names differ from the 08:15 list. If you already entered the 08:15 "
        f"list in the pre-open, keep it — do not double up.")
    push(f"🎯 FINAL gap-reversal shorts {session:%a %d %b}", body, enabled=args.push)
    _sync(f"chore(data): final picks for {session} [skip ci]")


def cmd_levels(args) -> None:
    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    picks = G.load_picks(session)
    if picks is None:
        print(f"no picks saved for {session}; run `picks` first")
        return
    symbols = [p.symbol for p in picks]
    if not args.no_fetch and G.missing_intraday(symbols, session):
        G.refresh(symbols, interval="5m", period="1d")
    opened = []
    from nse_intraday_ai import nse_bands
    try:
        bands = nse_bands.load(session, fetch=False)
    except Exception:                                         # noqa: BLE001 — optional input
        bands = nse_bands.parse("Symbol,Series,Security Name,Band,Remarks\n")
    capped: dict[str, float] = {}
    from nse_intraday_ai.nse_preopen import archived_opens

    official = archived_opens(session)          # archived by the 09:10 job: the official opens
    for p in picks:
        bars = G.load_session_bars(p.symbol, session)
        auction = official.get(p.symbol.removesuffix(".NS"))
        if auction or (not bars.empty and 0 in bars.index):
            p.entry = round(float(auction or bars.at[0, "open"]), 2)
            p.stop_price = round(p.entry + p.stop_distance, 2)
            cap = G.circuit_capped_stop(p, bands)
            if cap is not None:
                capped[p.symbol], p.stop_price = cap[1], cap[0]
            opened.append(p)
    if not opened:
        push(f"Gap-reversal {session:%d %b}", "No opening prices — market holiday or no data yet. "
             "No trades today.", enabled=args.push, priority="default")
        return
    previous = G.read_picks() or {}
    extra = {k: v for k, v in previous.items() if k not in ("session", "config", "picks")}
    G.save_picks(session, picks, CONFIG, **{**extra, "levels_at": datetime.now(IST).isoformat(timespec="seconds")})
    main = [p for p in opened if not p.reserve][:CONFIG.picks]
    body = "\n".join(
        f"{p.symbol.removesuffix('.NS'):<11} opened ₹{p.entry:,.2f} below limit ₹{p.limit_price:,.2f} "
        f"→ NOT FILLED, cancel the order"
        if p.limit_price and p.entry < p.limit_price else
        f"{p.symbol.removesuffix('.NS'):<11} open ₹{p.entry:,.2f} → BUY SL-M ₹{p.stop_price:,.2f}"
        + (f"  ⚠ capped below the ₹{capped[p.symbol]:,.2f} upper circuit" if p.symbol in capped else "")
        for p in main)
    body += (f"\nCover everything at {CONFIG.square_off}. (Stops use the "
             f"{'official auction open' if official else 'first 5-minute bar'}; use your own fill if different.)")
    push(f"🛑 Stop-loss levels {session:%d %b}", body, enabled=args.push)
    _sync(f"chore(data): stop levels for {session} [skip ci]")


def last_completed_session(now: datetime) -> date:
    day = now.date()
    if now.weekday() < 5 and now.strftime("%H:%M") >= "15:30":
        return day
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


class IncompleteSession(RuntimeError):
    """The intraday bars stop before the square-off: not recordable yet."""


def official_opens(session: date) -> dict[str, float]:
    """symbol -> the session's official open: the archived pre-open auction, else NSE's
    bhavcopy (published ~18:00), else {} (the first 5-minute bar's open is used)."""
    from nse_intraday_ai.nse_preopen import archived_opens

    opens = archived_opens(session)
    if opens:
        return opens
    try:
        from nse_intraday_ai import nse_bhav

        bhav = nse_bhav.load(session, session)
    except Exception:                                         # noqa: BLE001 — optional input
        return {}
    if bhav.empty:
        return {}
    eq = bhav[(bhav["series"] == "EQ") & (bhav["open"] > 0)]
    return dict(zip(eq["symbol"], eq["open"].astype(float)))


def record_session(session: date, *, fetch: bool, now: datetime | None = None) -> list[G.Trade]:
    """Paper-trade one session exactly as the backtest does; [] if no bars.

    Raises IncompleteSession if most picks' bars end before the 15:15 square-off
    of a finished session, so the day is retried instead of booked short.
    """
    picks = G.load_picks(session)
    if picks is None:
        # Nothing was published for this session (the laptop was off).  The
        # paper book records published lists only: rebuilding one with the
        # Yahoo gap rule would book a list the ranker-led system never traded.
        # The experts' grades for the day are in the meta state's track.
        return []
    symbols = [p.symbol for p in picks]
    if fetch and G.missing_intraday(symbols, session, bar=G.SQUARE_OFF_BAR):
        G.refresh(symbols, interval="5m", period="1mo")
    official = official_opens(session)          # the price a pre-open order fills at
    trades = []
    for p in picks:
        if len(trades) >= CONFIG.picks:
            break
        trade = G.simulate_pick(p, G.load_session_bars(p.symbol, session), session, CONFIG,
                                entry=official.get(p.symbol.removesuffix(".NS")))
        if trade is not None:
            trades.append(trade)
    now = now or datetime.now(IST)
    finished = session < now.date() or now.strftime("%H:%M") >= "15:30"
    short = [t for t in trades if t.exit_reason == "LAST_BAR"]
    if finished and trades and len(short) * 2 > len(trades):
        raise IncompleteSession(f"{len(short)}/{len(trades)} picks' bars end before 15:15 "
                                f"(last {max(t.exit_time for t in short)})")
    return trades


def cmd_record(args) -> None:
    wait_for_clock()
    book = pd.read_csv(PAPER) if PAPER.exists() else pd.DataFrame()
    if args.date:
        sessions = [date.fromisoformat(args.date)]
    else:
        # Every forward session not yet in the book (Yahoo keeps 60 days of 5m
        # bars), so an outage day is retried on the next run instead of being
        # skipped forever once a later day has been recorded.
        last = last_completed_session(datetime.now(IST))
        start = max(FORWARD_START, last - timedelta(days=45))
        done = set(book["session"]) if not book.empty else set()
        sessions = [d.date() for d in pd.bdate_range(start, last)
                    if d.date().isoformat() not in done]

    # Lists that went out after the entry cutoff were never tradable: take any
    # such session back out of the book (2026-10-14 and 10-16 had been booked).
    dropped_late = []
    if not book.empty:
        for s in sorted(set(book["session"])):
            payload = G.read_picks(G.dated_picks_path(date.fromisoformat(s)))
            if payload is not None and payload.get("session") == s and G.published_late(payload):
                dropped_late.append(s)
        if dropped_late:
            book = book[~book["session"].isin(dropped_late)]
            print(f"removed from the paper book (list published LATE): {', '.join(dropped_late)}")

    coverage = G.session_coverage(G.nifty500_symbols(), min(sessions)) if sessions else {}
    newest = max(coverage) if coverage else None
    recorded = []
    for session in sessions:
        if newest and session < newest and coverage.get(session, 0) == 0:
            continue                     # a holiday: no real bars, later days have them
        payload = G.picks_payload(session)
        if payload is None:
            print(f"{session}: no list was published — not recorded")
            continue
        if G.published_late(payload):
            print(f"{session}: the list went out after {G.ENTRY_CUTOFF} (LATE, 'skip today') — not recorded")
            continue
        try:
            trades = record_session(session, fetch=not args.no_fetch)
        except IncompleteSession as exc:
            print(f"{session}: intraday data incomplete ({exc}) — not recorded, retried next run")
            continue
        if not trades:
            print(f"{session}: no bars for any pick — holiday or data outage, nothing recorded")
            continue
        rows = pd.DataFrame([{**asdict(t), "net": round(t.net, 2)} for t in trades])
        rows["session"] = session.isoformat()
        if not book.empty:
            book = book[book["session"] != session.isoformat()]
        book = pd.concat([book, rows], ignore_index=True)
        recorded.append((session, trades))
    if not recorded and not dropped_late:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    book.sort_values(["session", "symbol"]).to_csv(PAPER, index=False)
    if not recorded:
        _sync("chore(data): paper book without LATE lists [skip ci]")
        return

    per_day = book.groupby("session")["net"].sum()
    for session, trades in recorded:
        net = sum(t.net for t in trades)
        body = "\n".join(f"{t.symbol.removesuffix('.NS'):<11} {t.exit_reason:<10} ₹{t.net:+,.0f}"
                         for t in trades)
        body += (f"\nDay: ₹{net:+,.0f} ({net / CONFIG.capital * 100:+.2f}%) · "
                 f"{sum(t.net > 0 for t in trades)}/{len(trades)} winners"
                 f"\nPaper book: {len(per_day)} sessions, ₹{per_day.sum():+,.0f} "
                 f"({per_day.sum() / CONFIG.capital * 100:+.2f}%), {(per_day > 0).sum()} up days")
        push(f"📒 Gap-reversal result {session:%d %b}", body,
             enabled=args.push and session == recorded[-1][0], priority="default")
    _sync(f"chore(data): paper result for {recorded[-1][0]} [skip ci]")


def _window_summary(result: G.BacktestResult) -> dict:
    return {"summary": result.summary(),
            "daily": [{**r, "session": str(r["session"])}
                      for r in result.daily().round(2).to_dict("records")]}


def cmd_backtest(args) -> None:
    symbols = G.nifty500_symbols()
    daily = G.load_daily(symbols, since="2016-01-01" if args.decade else "2026-03-01")
    sessions = [d for d in daily["close"].index if d >= date.fromisoformat(args.since)] \
        if args.since else list(daily["close"].index[-args.sessions:])
    if args.fetch:
        for s in sessions:
            need = G.missing_intraday([p.symbol for p in G.select(daily, s, CONFIG)], s,
                                      bar=G.SQUARE_OFF_BAR)
            if need:
                G.refresh(need, interval="5m", period="60d")
    result = G.backtest(sessions, CONFIG, daily=daily)
    s = result.summary()
    print(f"\nGap-reversal book, {s['sessions']} sessions {s['first']} .. {s['last']} "
          f"on ₹{CONFIG.capital:,.0f}")
    daily_pnl = result.daily()
    for row in daily_pnl.itertuples():
        day_trades = [t for t in result.trades if t.session == row.session]
        names = " ".join(f"{t.symbol.removesuffix('.NS')}({t.gross_bps:+.0f})" for t in day_trades)
        print(f"  {row.session}  ₹{row.net:+10,.0f}  cum ₹{row.cum_net:+11,.0f}  {names}")
    print(f"\n  net ₹{s['net_rupees']:+,.0f} ({s['net_pct']:+.2f}%) | {s['trades']} trades, "
          f"win {s['win_rate_pct']}%, PF {s['profit_factor']} | up/down days "
          f"{s['up_days']}/{s['down_days']} | max DD {s['max_drawdown_pct']}% | "
          f"gross {s['avg_gross_bps']:+.1f} bps, net {s['avg_net_bps']:+.1f} bps/trade | "
          f"costs ₹{s['costs_rupees']:,.0f}")

    payload = {"generated_at": datetime.now(IST).isoformat(timespec="seconds"),
               "config": asdict(CONFIG), "window": _window_summary(result)}
    if args.decade:
        d = G.backtest_daily(daily, CONFIG)
        x = d["net_bps"]
        years = x.groupby([s.year for s in x.index]).mean().round(1)
        eq = (x / 100).cumsum()
        payload["decade"] = {
            "sessions": int(len(x)), "first": str(x.index[0]), "last": str(x.index[-1]),
            "net_bps_per_trade": round(float(x.mean()), 2),
            "t_stat": round(float(x.mean() / x.std() * len(x) ** 0.5), 2),
            "up_day_pct": round(float((x > 0).mean() * 100), 1),
            "max_drawdown_pct": round(float((eq.cummax() - eq).max()), 2),
            "by_year": {int(k): float(v) for k, v in years.items()},
        }
        dd = payload["decade"]
        print(f"\nDecade check (daily bars, open->close, 13 bps cost): {dd['sessions']} sessions, "
              f"{dd['net_bps_per_trade']:+.1f} bps/trade net, t={dd['t_stat']}, "
              f"max DD {dd['max_drawdown_pct']}%")
        print("  " + " ".join(f"{y}:{v:+.1f}" for y, v in dd["by_year"].items()))
    if args.save:
        from nse_intraday_ai.atomic_io import atomic_write_json
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(BACKTEST, payload)
        print(f"\n-> {BACKTEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("picks", "final", "levels", "record"):
        p = sub.add_parser(name)
        p.add_argument("--date", help="ISO session date (default: the relevant session)")
        p.add_argument("--push", action="store_true", help="send to the phone via ntfy")
        p.add_argument("--no-fetch", action="store_true", help="use the candle cache as-is")
        if name == "picks":
            p.add_argument("--history-days", type=int, default=HISTORY_DAYS_DEFAULT,
                           help="calendar days of NSE files to refresh (45 on a fresh machine)")
    f = sub.add_parser("cloud-fallback", help="GitHub Actions: gap-rule list by 08:21 IST unless the laptop "
                                               "publishes first")
    f.add_argument("--push", action="store_true", help="send to the phone via ntfy when published")
    f.add_argument("--no-fetch", action="store_true", help="use the local cache as-is")
    f.add_argument("--history-days", type=int, default=45, help="calendar days of NSE files to refresh")
    f.add_argument("--build-at", default=FALLBACK_BUILD_AT, help="HH:MM IST to start building")
    f.add_argument("--deadline", default=FALLBACK_DEADLINE, help="HH:MM IST the laptop has priority until")
    f.add_argument("--dry-run", action="store_true", help="build and decide, but neither publish nor push")
    f.add_argument("--any-day", action="store_true", help="skip the trading-day check (tests/dry runs)")
    b = sub.add_parser("backtest")
    b.add_argument("--sessions", type=int, default=14, help="replay the last N sessions")
    b.add_argument("--since", help="replay every session from this ISO date instead")
    b.add_argument("--decade", action="store_true", help="add the 10-year daily-bar check")
    b.add_argument("--fetch", action="store_true", help="download missing 5m bars (60-day limit)")
    b.add_argument("--save", action="store_true", help=f"write {BACKTEST.relative_to(ROOT)} for the app")
    args = parser.parse_args()
    {"picks": cmd_picks, "final": cmd_final, "levels": cmd_levels, "record": cmd_record,
     "backtest": cmd_backtest, "cloud-fallback": cmd_cloud_fallback}[args.cmd](args)


if __name__ == "__main__":
    main()
