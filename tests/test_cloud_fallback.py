"""Cloud fallback (GitHub Actions): timing, stand-aside rules, race safety, schedule.

All times are IST.  A fake clock drives `cmd_cloud_fallback`, so a whole
03:00 → 08:21 morning runs in milliseconds.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
IST = ZoneInfo("Asia/Kolkata")
MON = date(2026, 10, 12)


def _gs():
    spec = importlib.util.spec_from_file_location("gap_script_fb", ROOT / "scripts" / "gap_reversal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    def __init__(self, hhmm: str, day: date = MON):
        h, m = map(int, hhmm.split(":"))
        self.now = datetime(day.year, day.month, day.day, h, m, tzinfo=IST)
        self.slept = 0.0

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.slept += seconds
        self.now += timedelta(seconds=seconds)

    def at(self, hhmm: str) -> datetime:
        h, m = map(int, hhmm.split(":"))
        return self.now.replace(hour=h, minute=m, second=0, microsecond=0)


def _list(publisher: str, at: datetime, session: date = MON) -> dict:
    return {"session": session.isoformat(), "generated_at": at.isoformat(timespec="seconds"),
            "publisher": publisher, "picks": []}


def _args(**kw):
    base = dict(push=True, no_fetch=True, history_days=45, build_at="08:00", deadline="08:20",
                dry_run=False, any_day=False)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def env(monkeypatch):
    """The script with the build, the calendar and the phone faked; records what happened."""
    gs = _gs()
    G = gs.G
    rec = SimpleNamespace(built=[], published=[], pushed=[], remote=lambda now: None)
    picks = [G.Pick(rank=i, symbol=f"S{i}.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0, atr=2.0,
                    stop_distance=1.5, quantity=1000, position_value=1e5, turnover_cr=50.0, reserve=i > 8)
             for i in range(1, 11)]

    def fake_publish_list(target, fetch, history_days):
        rec.built.append(rec.clock())
        rec.clock.sleep(95)                         # a fresh runner: 45 days of NSE files, ~1.5 min
        return target, target - timedelta(days=3), picks, {"ranked_by": "rule"}

    monkeypatch.setattr(gs, "_publish", fake_publish_list)
    monkeypatch.setattr(gs, "push", lambda title, body, enabled, priority="high":
                        rec.pushed.append((rec.clock(), title, enabled)))
    import nse_intraday_ai.nse_calendar as cal
    monkeypatch.setattr(cal, "is_trading_day", lambda d: d.weekday() < 5 and d != date(2026, 10, 20))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def run(start: str, day: date = MON, ok: bool = True, **kw):
        rec.clock = Clock(start, day)
        return gs.cmd_cloud_fallback(
            _args(**kw), clock=rec.clock, sleep=rec.clock.sleep, remote=lambda: rec.remote(rec.clock()),
            publish=lambda msg: (rec.published.append((rec.clock(), msg)) or ok))

    rec.run, rec.gs = run, gs
    return rec


def test_laptop_off_main_run_waits_from_0300_and_publishes_before_0830(env):
    assert env.run("03:00") == "published"
    assert env.built[0] == env.clock.at("08:00")                    # slept ~5 h, then built
    when, msg = env.published[0]
    assert env.clock.at("08:20") <= when < env.clock.at("08:22") < env.clock.at("08:30")
    assert "cloud-fallback picks for 2026-10-12" in msg
    t, title, enabled = env.pushed[0]
    assert t >= when and enabled and "Cloud-fallback" in title     # phone only after main has it


def test_laptop_list_at_0816_wins_and_nothing_is_published(env):
    env.remote = lambda now: _list("laptop", now) if now >= now.replace(hour=8, minute=16) else None
    assert env.run("03:00") == "stood aside: laptop"
    assert env.published == [] and env.pushed == []
    assert env.clock() < env.clock.at("08:17")                      # noticed within one poll


def test_laptop_list_already_there_skips_the_build(env):
    env.remote = lambda now: _list("laptop", now.replace(hour=8, minute=16))
    assert env.run("08:24") == "stood aside: laptop"
    assert env.built == [] and env.published == []


def test_backup_run_stands_aside_after_the_main_run_published(env):
    env.remote = lambda now: _list("github-actions", now.replace(hour=8, minute=20))
    assert env.run("08:22") == "stood aside: github-actions"
    assert env.built == [] and env.pushed == []


def test_backup_that_github_started_late_still_publishes_before_0830(env):
    assert env.run("08:25") == "published"                          # main run was dropped
    assert env.published[0][0] < env.clock.at("08:30")


def test_last_evenings_preliminary_list_does_not_count(env):
    sunday = datetime(2026, 10, 11, 21, 0, tzinfo=IST)
    env.remote = lambda now: _list("laptop", sunday)                # session Mon, built Sun
    assert env.run("03:00") == "published"


def test_exchange_holiday_and_weekend_do_nothing(env):
    assert env.run("03:00", day=date(2026, 10, 20)) == "holiday"
    assert env.run("03:00", day=date(2026, 10, 17)) == "holiday"
    assert env.built == [] and env.published == []


def test_losing_the_push_race_to_the_laptop_is_not_an_error(env):
    state = {"laptop": False}

    def remote(now):
        return _list("laptop", now) if state["laptop"] else None

    env.remote = remote

    def lost(msg):          # the guard saw the laptop's commit on main
        state["laptop"] = True
        return False

    env.clock = Clock("03:00")
    out = env.gs.cmd_cloud_fallback(_args(), clock=env.clock, sleep=env.clock.sleep,
                                    remote=lambda: env.remote(env.clock()), publish=lost)
    assert out == "stood aside: laptop" and env.pushed == []


def test_failed_publish_with_nobody_else_fails_the_job(env):
    with pytest.raises(SystemExit):
        env.run("03:00", ok=False)
    assert env.pushed == []


def test_dry_run_builds_but_never_publishes_or_pushes(env):
    assert env.run("13:00", day=date(2026, 10, 11), dry_run=True, any_day=True,
                   build_at="00:00", deadline="00:00") == "dry run"
    assert env.built and env.published == [] and env.pushed == []


def test_published_today_rules():
    gs = _gs()
    at = datetime(2026, 10, 12, 8, 16, tzinfo=IST)
    assert gs.published_today(_list("laptop", at), MON) == "laptop"
    assert gs.published_today(_list("github-actions", at), MON) == "github-actions"
    assert gs.published_today(_list("laptop", at - timedelta(days=1)), MON) is None   # evening list
    assert gs.published_today(_list("laptop", at, session=date(2026, 10, 13)), MON) is None
    assert gs.published_today(None, MON) is None
    legacy = {"session": "2026-10-12", "generated_at": at.isoformat()}
    assert gs.published_today({**legacy, "model": {"x": 1}}, MON) == "laptop"
    assert gs.published_today(legacy, MON) == "github-actions"
    assert gs.laptop_already_published(_list("laptop", at), MON)


def test_laptop_message_says_it_replaces_a_fallback_list(monkeypatch):
    gs = _gs()
    monkeypatch.setattr(gs, "_remote_picks",
                        lambda: _list("github-actions", datetime(2026, 10, 12, 8, 20, tzinfo=IST)))
    assert gs._replaces_fallback_note(MON).startswith("REPLACES the cloud-fallback list sent at 08:20")
    monkeypatch.setattr(gs, "_remote_picks", lambda: _list("laptop", datetime(2026, 10, 12, 8, 16, tzinfo=IST)))
    assert gs._replaces_fallback_note(MON) == ""


def test_cli_parses():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gap_reversal.py"), "cloud-fallback", "--help"],
                         cwd=ROOT, capture_output=True, text=True, timeout=120,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert out.returncode == 0 and "--deadline" in out.stdout, out.stderr[-2000:]


# ── the workflow file ───────────────────────────────────────────────────────

def _ist(cron: str) -> tuple[int, int, set[int]]:
    """(hour, minute, ISO weekdays 1-7) in IST of a 'M H * * D' UTC cron with a D range."""
    m, h, _, _, dow = cron.split()
    lo, hi = map(int, dow.split("-"))
    total = int(h) * 60 + int(m) + 330
    shift = total // (24 * 60)
    days = {((d + shift - 1) % 7) + 1 for d in range(lo, hi + 1)}     # cron 0=Sun → ISO 7
    days = {7 if d == 0 else d for d in days}
    return (total % 1440) // 60, total % 60, days


def test_workflow_schedule_meets_the_0830_ist_deadline():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "gap_picks.yml").read_text())
    on = wf[True] if True in wf else wf["on"]
    crons = [_ist(c["cron"]) for c in on["schedule"]]
    weekdays = {1, 2, 3, 4, 5}
    assert all(days == weekdays for *_, days in crons), crons            # Mon-Fri IST only
    main = [(h, m) for h, m, _ in crons if h < 7]
    backups = [(h, m) for h, m, _ in crons if h >= 7]
    assert main, "an early run that waits for 08:00 IST is required (GitHub starts runs late)"
    assert all((7, 30) <= b <= (8, 15) for b in backups), backups
    job = wf["jobs"]["morning_picks"]
    h, m = main[0]
    needed = (8 * 60 + 22) - (h * 60 + m)                                # until ~08:22 IST
    assert job["timeout-minutes"] >= needed + 10
    assert job["timeout-minutes"] <= 360                                 # GitHub's hard limit
    assert wf["concurrency"]["cancel-in-progress"] is False             # never kill a waiting run
    run = job["steps"][-1]["run"]
    assert "cloud-fallback --push" in run and "--dry-run" in run
    assert on["workflow_dispatch"]["inputs"]["mode"]["default"] == "dry-run"
    assert job["permissions"]["contents"] == "write"
