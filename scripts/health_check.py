"""One command that answers "is this thing actually working?".

Originally written after 2026-08-14, when three failures had been running silently:

  * the Streamlit process had leaked to 15.2 GB over 35 days, stopped answering
    HTTP, and filled swap;
  * because swap was full, the (since retired) paper book's run took over 5
    minutes and systemd killed it mid-flight every tick — a whole session
    recorded nothing;
  * the macro context symbols had gone stale in the candle cache, so the gate
    refused *every* signal for two sessions and the screen simply showed
    nothing, which looks identical to a quiet market.

None of those raised an alarm. Each is trivially detectable. This checks them.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --quiet   # only print problems (for cron)
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "candles.sqlite3"

OK, WARN, FAIL = "ok", "warn", "FAIL"
# Every timer in deploy/systemd plus the app: one source of truth with install.sh.
UNITS = ["nse-signal-lab.service"] + sorted(
    p.name for p in (ROOT / "deploy" / "systemd").glob("*.timer"))
sys.path.insert(0, str(ROOT / "src"))


def _run(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def check_units() -> list[tuple[str, str, str]]:
    out = []
    for unit in UNITS:
        state = _run("systemctl", "--user", "is-active", unit) or "unknown"
        status = OK if state in ("active", "running", "waiting") else FAIL
        out.append((f"unit {unit}", status, state))
    linger = "yes" in _run("loginctl", "show-user", str(Path.home().name), "-p", "Linger").lower()
    out.append((
        "survives logout (linger)", OK if linger else WARN,
        "enabled" if linger else "disabled — services die when you log out",
    ))
    return out


def _service_age_s(unit: str) -> float | None:
    """Seconds since `unit` became active (None if unknown)."""
    raw = _run("systemctl", "--user", "show", unit, "-p", "ActiveEnterTimestampMonotonic", "--value")
    try:
        since = int(raw) / 1e6
        return float(Path("/proc/uptime").read_text().split()[0]) - since if since else None
    except (ValueError, OSError):
        return None


def check_app() -> tuple[str, str, str]:
    code = _run("curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                "--max-time", "20", "http://127.0.0.1:8501/")
    if code != "200":
        # At boot this check runs while Streamlit is still starting (2026-10-06 20:30: HTTP 000).
        age = _service_age_s("nse-signal-lab.service")
        if age is not None and age < 180:
            return ("webpage http://127.0.0.1:8501", WARN, f"HTTP {code or 'no response'} (app started {age:.0f}s ago)")
    return ("webpage http://127.0.0.1:8501",
            OK if code == "200" else FAIL,
            f"HTTP {code or 'no response'}")


def check_memory() -> tuple[str, str, str]:
    """The leak that started all of this."""
    pid = _run("systemctl", "--user", "show", "nse-signal-lab.service", "-p", "MainPID")
    pid = pid.split("=")[-1] if "=" in pid else ""
    if not pid or pid == "0":
        return ("app memory", FAIL, "not running")
    try:
        rss_kb = int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0])
    except Exception:
        return ("app memory", WARN, "unreadable")
    gb = rss_kb / 1024 / 1024
    status = OK if gb < 3 else (WARN if gb < 5 else FAIL)
    return ("app memory", status, f"{gb:.1f} GB (cap 5 GB, nightly restart)")


def _prev_trading_day(day):
    from nse_intraday_ai.nse_calendar import is_trading_day

    d = day - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def check_data(now: datetime) -> list[tuple[str, str, str]]:
    """The inputs the 08:45 list is built from must cover the last session."""
    out = []
    files = sorted((ROOT / "data" / "nse_bhav").glob("*/bhav_*.parquet"))
    latest = datetime.strptime(files[-1].stem.split("_")[1], "%Y%m%d").date() if files else None
    # NSE publishes the bhavcopy ~18:00; the morning job fetches it at 08:45.
    need = _prev_trading_day(now.date()) if now.strftime("%H:%M") >= "08:50" else \
        _prev_trading_day(_prev_trading_day(now.date()))
    status = OK if latest and latest >= need else FAIL
    out.append(("NSE bhavcopy", status, f"latest {latest}, need {need}"))
    if DB.exists():
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(ts) FROM candles WHERE symbol='^NSEI' AND interval='1d'").fetchone()
        finally:
            con.close()
        last = datetime.fromisoformat(row[0]).astimezone(IST).date() if row and row[0] else None
        # The Yahoo fallback downloads its own bars when it runs, so a stale cache is not a fault.
        out.append(("daily candles (fallback rule)", OK, f"^NSEI 1d through {last}; refreshed on use"))
    else:
        out.append(("daily candles (fallback rule)", WARN, "candle cache missing"))
    return out


FORWARD_START = "2026-09-29"            # first session of the live paper record


def _sessions(since: str, through: str) -> list[str]:
    """NSE sessions in [since, through], from the bhavcopy files (holidays have none)."""
    days = (f.stem.split("_")[1] for f in (ROOT / "data" / "nse_bhav").glob("*/bhav_*.parquet"))
    days = sorted(f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in days)
    return [d for d in days if since <= d <= through]


def check_learning() -> tuple[str, str, str]:
    """Every session since the live record began is graded in the meta state.

    2026-10-06: after a day off, the catch-up graded only the newest session and
    2026-10-05 never entered the guard's paired record.
    """
    import json

    path = ROOT / "data" / "models" / "meta_state.json"
    if not path.exists():
        return ("learning record", WARN, "no meta state yet")
    st = json.loads(path.read_text())
    through = st.get("updated_through") or FORWARD_START
    graded = {r["session"] for r in st.get("track", [])}
    missing = [d for d in _sessions(FORWARD_START, through) if d not in graded]
    return ("learning record", WARN if missing else OK,
            f"sessions never graded: {', '.join(missing)}" if missing else f"complete through {through}")


def check_preopen(now: datetime) -> tuple[str, str, str]:
    """The 09:10 job archives every auction; the archive must grow and match the official opens.

    Until 2026-10-06 a wrong finality test rejected every auction that closed
    before 09:10, and the archive stayed empty for weeks without a warning.
    """
    import pandas as pd

    files = sorted((ROOT / "data" / "nse_preopen").glob("*/preopen_*.parquet"))
    if not files:
        return ("pre-open archive", WARN, "empty — the 09:10 job has archived no auction")
    day = files[-1].stem.split("_")[1]
    detail = f"{len(files)} sessions, latest {day[:4]}-{day[4:6]}-{day[6:]}"
    recent = _sessions(FORWARD_START, now.date().isoformat())[-5:]
    stale = len(recent) == 5 and day < recent[0].replace("-", "")     # none of the last 5 sessions
    bhav = ROOT / "data" / "nse_bhav" / day[:4] / f"bhav_{day}.parquet"
    if bhav.exists():
        auction = pd.read_parquet(files[-1])
        official = pd.read_parquet(bhav, columns=["symbol", "series", "open"]).query("series == 'EQ'")
        j = auction.merge(official, on="symbol")
        if len(j):
            match = float(((j["iep"] / j["open"] - 1).abs() <= 1e-4).mean())
            detail += f"; {match:.0%} equal the official open"
            if match < 0.95:
                return ("pre-open archive", WARN, detail)
    return ("pre-open archive", WARN if stale else OK,
            detail + ("; nothing archived in the last 5 sessions" if stale else ""))


def check_published(now: datetime, last_n: int = 5) -> tuple[str, str, str]:
    """Did each recent session get a list?  (None is published while the laptop is off.)"""
    import csv

    book = ROOT / "data" / "gap_reversal"
    recorded = set()
    if (book / "paper_book.csv").exists():
        with (book / "paper_book.csv").open() as handle:
            recorded = {r["session"] for r in csv.DictReader(handle)}
    recent = _sessions(FORWARD_START, now.date().isoformat())[-last_n:]
    missed = [d for d in recent if d not in recorded
              and not (book / "history" / f"picks_{d.replace('-', '')}.json").exists()]
    return ("lists published", WARN if missed else OK,
            f"no list for {', '.join(missed)} (laptop off?)" if missed else f"last {len(recent)} sessions covered")


def market_is_open(now: datetime) -> bool:
    from nse_intraday_ai.nse_calendar import is_trading_day

    return is_trading_day(now.date()) and "09:15" <= now.strftime("%H:%M") <= "15:30"


def check_book(now: datetime) -> list[tuple[str, str, str]]:
    """The validated book: is today's pick list out, and is the record current?"""
    import csv
    import json

    from nse_intraday_ai.nse_calendar import is_trading_day

    out = []
    book = ROOT / "data" / "gap_reversal"
    picks = book / "picks.json"
    if not picks.exists():
        out.append(("gap-reversal picks", FAIL, "data/gap_reversal/picks.json missing"))
    else:
        session = json.loads(picks.read_text()).get("session", "?")
        # From 08:45 on a trading day the list must be for today.
        due = is_trading_day(now.date()) and now.strftime("%H:%M") >= "08:50"
        stale = due and now.strftime("%H:%M") <= "15:30" and session < now.date().isoformat()
        out.append(("gap-reversal picks", FAIL if stale else OK, f"list for {session}"))

    if picks.exists():
        blank = json.loads(picks.read_text()).get("missing_inputs") or []
        out.append(("model inputs", WARN if blank else OK,
                    f"{len(blank)} blank for every candidate: {', '.join(blank[:6])}" if blank
                    else "all sources present in the latest list"))

    report = ROOT / "data" / "models" / "weekly_report.json"
    if report.exists():
        r = json.loads(report.read_text())
        age = (now.date() - datetime.fromisoformat(r["at"]).date()).days
        out.append(("model review", OK if age <= 9 else WARN,
                    f"{r['at'][:10]} ({age}d ago), {'promoted' if r['promoted'] else 'kept previous'}"))
    else:
        out.append(("weekly model review", WARN, "never run (python scripts/learn.py)"))
    state = ROOT / "data" / "models" / "meta_state.json"
    if state.exists():
        st = json.loads(state.read_text())
        out.append(("drift alarm", FAIL if st.get("alarm") else OK,
                    f"CUSUM {st.get('cusum', 0):.0f}/{st.get('threshold_bps')} bps, "
                    f"learned through {st.get('updated_through')}"))
    record = book / "paper_book.csv"
    if not record.exists():
        out.append(("gap-reversal paper book", WARN, "no session recorded yet"))
        return out
    with record.open() as handle:
        rows = list(csv.DictReader(handle))
    sessions = sorted({r["session"] for r in rows})
    total = sum(float(r["net"]) for r in rows)
    # After 15:45 on a trading day that session must be recorded.
    expected = now.date() if is_trading_day(now.date()) and now.strftime("%H:%M") >= "15:45" else None
    status = WARN if expected and sessions[-1] != expected.isoformat() else OK
    out.append(("gap-reversal paper book", status,
                f"{len(sessions)} sessions, latest {sessions[-1]}, ₹{total:+,.0f} cumulative"))
    return out


def check_intraday(now: datetime) -> tuple[str, str, str]:
    """During the session the 5-minute monitor feed must be fresh (< 15 min old)."""
    import json

    from nse_intraday_ai.nse_calendar import is_trading_day

    live = ROOT / "data" / "intraday" / "live.json"
    due = is_trading_day(now.date()) and "09:35" <= now.strftime("%H:%M") <= "15:45"
    if not live.exists():
        return ("intraday monitor feed", WARN if due else OK, "data/intraday/live.json not written yet")
    try:
        gen = datetime.fromisoformat(json.loads(live.read_text())["generated_at"])
    except (ValueError, KeyError, TypeError):
        return ("intraday monitor feed", WARN, "live.json unreadable")
    age = (now - gen).total_seconds() / 60
    return ("intraday monitor feed", WARN if due and age > 15 else OK,
            f"last scan {gen:%Y-%m-%d %H:%M} ({age:.0f} min ago)")


def check_disk() -> tuple[str, str, str]:
    out = _run("df", "-h", str(ROOT))
    line = out.splitlines()[-1] if out else ""
    parts = line.split()
    if len(parts) < 5:
        return ("disk", WARN, "unreadable")
    used = int(parts[4].rstrip("%"))
    return ("disk", OK if used < 90 else WARN, f"{parts[4]} used, {parts[3]} free")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print only problems")
    args = parser.parse_args()

    now = datetime.now(tz=IST)
    checks: list[tuple[str, str, str]] = []
    checks += check_units()
    checks.append(check_app())
    checks.append(check_memory())
    checks.append(check_disk())
    checks += check_data(now)
    checks += check_book(now)
    checks.append(check_learning())
    checks.append(check_published(now))
    checks.append(check_preopen(now))
    checks.append(check_intraday(now))

    problems = [c for c in checks if c[1] != OK]
    width = max(len(name) for name, _, _ in checks)
    print(f"NSE Gap-Reversal Book health — {now:%Y-%m-%d %H:%M %Z}"
          f"  (market {'OPEN' if market_is_open(now) else 'closed'})")
    print("-" * (width + 34))
    for name, status, detail in checks:
        if args.quiet and status == OK:
            continue
        mark = {OK: "  ok ", WARN: " warn", FAIL: " FAIL"}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    if not problems:
        print("\nall good.")
        return 0
    print(f"\n{len(problems)} problem(s). Common fixes:")
    print("  stale bhavcopy : python scripts/gap_reversal.py picks   (fetches NSE data)")
    print("  app not serving: systemctl --user restart nse-signal-lab.service")
    print("  no pick list   : python scripts/gap_reversal.py picks")
    print("  missing session: python scripts/gap_reversal.py record --date YYYY-MM-DD")
    # Warnings are reported but do not fail the unit; only real failures do.
    return 1 if any(c[1] == FAIL for c in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
