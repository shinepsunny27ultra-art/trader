#!/usr/bin/env python3
"""Keep the laptop awake through a trading window — on mains power only.

    python scripts/keep_awake.py 1615 --trading-days       # 07:55 timer: through today's session
    python scripts/keep_awake.py 2110 --next-session 1615  # 20:50 timer: through the night and
                                                           # tomorrow's session if it is one

2026-10-27: the laptop slept through the session (lid closed after the 08:15 list),
woke at 19:15 with no network, and the day was booked on truncated data; at 20:43
it went to sleep again and the 21:00 evening list ran at 21:14.  Run by
nse-keep-awake-morning.timer (trading days 06:00, or switch-on, → 16:15: list, auction, stops,
intraday alerts, record) and nse-keep-awake-evening.timer (20:50 → the evening
list, then on through the night to 16:15 when tomorrow is a session: a laptop
asleep at 08:15 cannot wake itself for the list).  The lock ("NSE gap book" in
`systemd-inhibit --list`) is held only while the laptop is on mains power and is
released the moment it is unplugged, so a laptop closed into a bag on battery
still sleeps; plugged back in, the service takes it again within a minute
(exit 75 = "retry": Restart=on-failure).  Suspend from the menu then
asks to override it; `systemctl --user stop nse-keep-awake-morning nse-keep-awake-evening`
releases it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

POWER = Path("/sys/class/power_supply")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def on_mains(power: Path = POWER) -> bool:
    """True if any mains/USB-C adapter reports `online`.  No power_supply info: assume mains."""
    flags = [p.read_text().strip() for p in power.glob("*/online") if p.is_file()]
    return not flags or "1" in flags


RETRY = 75                     # EX_TEMPFAIL: on battery / unplugged — systemd restarts the unit


def until_today(hhmm: str, now: datetime) -> datetime:
    h, m = int(hhmm[:2]), int(hhmm[-2:])
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


def window_end(until: str, now: datetime, *, next_session: str | None = None,
               is_trading_day=None) -> datetime:
    """`until` today; with `next_session`, that time tomorrow if tomorrow is an NSE session."""
    end = until_today(until, now)
    if next_session:
        from datetime import timedelta

        if is_trading_day is None:
            from nse_intraday_ai.nse_calendar import is_trading_day
        tomorrow = now + timedelta(days=1)
        if is_trading_day(tomorrow.date()):
            end = until_today(next_session, tomorrow)
    return end


def hold(end: datetime, *, clock=datetime.now, sleep=time.sleep, mains=on_mains, poll: float = 30.0) -> str:
    """Block (inside systemd-inhibit) until `end` or until the laptop is unplugged."""
    while clock() < end:
        if not mains():
            return "unplugged"
        sleep(poll)
    return "window over"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("until", help="end of the window, HHMM (today)")
    ap.add_argument("--trading-days", action="store_true", help="hold only on NSE trading days")
    ap.add_argument("--next-session", metavar="HHMM",
                    help="if tomorrow is an NSE session, hold through the night to HHMM tomorrow")
    ap.add_argument("--hold-until", help=argparse.SUPPRESS)   # the process inside the lock (ISO time)
    args = ap.parse_args()
    if args.hold_until:
        outcome = hold(datetime.fromisoformat(args.hold_until))
        print(f"awake until {args.hold_until[:16]}: {outcome}", flush=True)
        return RETRY if outcome == "unplugged" else 0
    end = window_end(args.until, datetime.now(), next_session=args.next_session)
    if datetime.now() >= end:
        print(f"window ended at {end:%H:%M}; nothing to hold")
        return 0
    if args.trading_days:
        from nse_intraday_ai.nse_calendar import is_trading_day
        if not is_trading_day(datetime.now().date()):
            print("not an NSE trading day; nothing to hold")
            return 0
    if not on_mains():
        print("on battery: not holding off sleep (retried when plugged in)")
        return RETRY
    os.execvp("systemd-inhibit", [
        "systemd-inhibit", "--what=sleep:handle-lid-switch", "--mode=block", "--who=NSE gap book",
        f"--why=Trading window until {end:%a %H:%M} (released if unplugged)",
        sys.executable, os.path.abspath(__file__), args.until, "--hold-until", end.isoformat()])
    return 1                                   # not reached


if __name__ == "__main__":
    sys.exit(main())
