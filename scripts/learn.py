"""Weekly self-improvement: retrain the ranker, re-validate it, promote only on evidence.

    python scripts/learn.py                      # full cycle (~25 min, ~8 GB RAM)
    python scripts/learn.py --push               # and send the verdict to the phone
    python scripts/learn.py --if-needed --push   # the timer: Saturdays, on a drift alarm,
                                                 # or if no review has ever run

Steps: refresh the NSE/F&O/global data, rebuild the point-in-time history,
walk-forward the ranker over the last two years (quarterly refits), compare its
out-of-sample book with the gap rule's, fit a challenger on everything, and
promote it only if it beat the rule with t >= 2.  The verdict and the evidence
are written to data/models/weekly_report.json for the app.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai import pipeline as P  # noqa: E402


def wait_for_todays_list(now=None, *, deadline: str = "08:30", poll: float = 30.0, sleep=None) -> bool:
    """On a trading-day morning, let the 08:15 list go out before a retrain starts.

    The retrain timer is persistent, so a laptop switched on at 08:05 would start
    a 10 GB, ~25-minute retrain right away, competing with the list for CPU and
    memory.  Returns True if it waited for (and saw) today's list.
    """
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from nse_intraday_ai import gap_reversal as G
    from nse_intraday_ai.nse_calendar import is_trading_day

    ist = ZoneInfo("Asia/Kolkata")
    clock = now or (lambda: datetime.now(ist))
    sleep = sleep or time.sleep
    today = clock().date()
    if not is_trading_day(today) or not "05:00" <= clock().strftime("%H:%M") < deadline:
        return False
    print(f"waiting for today's 08:15 list (until {deadline}) before retraining", flush=True)
    while clock().strftime("%H:%M") < deadline:
        payload = G.read_picks()
        if payload and payload.get("session") == today.isoformat() and payload.get("published_at"):
            return True
        sleep(poll)
    return False


def grade_before_retrain() -> None:
    """Fold every finished, unlearned session into the guard with the outgoing models.

    A challenger is trained on everything through today, and `learn_from` will
    not let a model grade a session it was trained on.  Saturday retrains lost
    nothing; an evening retrain after a session (any weekday, since retrains
    follow the review's age) would drop that day's ranker grade from the paired
    record.  2026-10-27: the day's -108 bps ranker book was nearly lost.
    """
    from nse_intraday_ai import meta as M
    from nse_intraday_ai import ranker as R

    try:
        through = P.previous_session(date.today() + timedelta(days=1))
    except P.DataNotReady:
        try:
            through = P.previous_session(date.today())
        except P.DataNotReady as exc:
            print(f"grading skipped: {exc}")
            return
    state = M.MetaState.load()
    if P.learn_through(through, state, R.load_champion(), open_model=R.load_champion("ranker_open"),
                       log=lambda m: print(m, flush=True)):
        state.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--if-needed", action="store_true",
                        help="only retrain on Saturdays, when the drift alarm is up, or when the "
                             "last review is 7+ days old (the timer runs this every evening)")
    args = parser.parse_args()

    if args.if_needed:
        from nse_intraday_ai import meta as M
        from nse_intraday_ai import ranker as R

        reasons = []
        if date.today().weekday() == 5:
            reasons.append("weekly schedule")
        # A Saturday the laptop was off must not skip the week: by 2026-10-27 the
        # champion was 11 days old because no Saturday 19:00 had found it on.
        review = R.MODELS / "weekly_report.json"
        if review.exists():
            import json
            from datetime import datetime

            age = (date.today() - datetime.fromisoformat(json.loads(review.read_text())["at"]).date()).days
            if age >= 7:
                reasons.append(f"last review {age} days ago")
        if M.MetaState.load().alarm:
            reasons.append("drift alarm")
        if not (R.MODELS / "weekly_report.json").exists():
            reasons.append("never reviewed")
        if not reasons:
            print("no retrain needed today")
            return
        print("retraining: " + ", ".join(reasons))

    wait_for_todays_list()
    if not args.no_fetch:
        P.refresh(date.today() - timedelta(days=1), days=30)
    grade_before_retrain()
    report = P.weekly(log=lambda m: print(m, flush=True))
    r, b, o = report["ranker"], report["rule"], report["ranker_open"]
    v, vo = report["vs_rule"], report["open_vs_ranker"]
    verdict = lambda n: "PROMOTED" if report["verdicts"][n]["promoted"] else "kept previous"  # noqa: E731
    body = (f"Walk-forward {report['window'][0]}..{report['window'][1]} (net bps/trade):\n"
            f"rule {b.get('net_bps_per_trade', float('nan')):+.1f} | 08:15 ranker "
            f"{r.get('net_bps_per_trade', float('nan')):+.1f} | 09:09 open ranker "
            f"{o.get('net_bps_per_trade', float('nan')):+.1f}\n"
            f"ranker - rule {v['mean_diff_bps']:+.1f}/day (t {v['t_stat']}): {verdict('ranker')}\n"
            f"open - ranker {vo['mean_diff_bps']:+.1f}/day (t {vo['t_stat']}): {verdict('ranker_open')}")
    print(body)
    from nse_intraday_ai import cloud_sync

    cloud_sync.publish(cloud_sync.state_files(), f"chore(data): weekly model review {date.today()} [skip ci]")
    if args.push:
        from nse_intraday_ai.alerts import send_ntfy
        send_ntfy(body, title="🧠 Weekly model review", priority="default", tags="brain")


if __name__ == "__main__":
    main()
