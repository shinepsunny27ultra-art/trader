"""Script-level smoke tests: every CLI parses, and the 08:15 message renders.

A SyntaxError in scripts/gap_reversal.py once slipped past the unit tests
(they only import src/); these tests exercise the scripts themselves.
"""
from __future__ import annotations

import importlib.util
import py_compile
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
# daily_monitor writes a report when run; everything else must be side-effect free on --help.
HELP_SAFE = [p for p in SCRIPTS if p.name != "daily_monitor.py"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", HELP_SAFE, ids=lambda p: p.name)
def test_script_help(path):
    out = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True,
                         text=True, timeout=120, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin",
                                                      "HOME": str(Path.home())})
    assert out.returncode == 0, out.stderr[-2000:]


def _load_gap_script():
    spec = importlib.util.spec_from_file_location("gap_script", ROOT / "scripts" / "gap_reversal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_picks_message_renders_limit_orders(monkeypatch, capsys):
    gs = _load_gap_script()
    G = gs.G
    picks = [G.Pick(rank=i, symbol=f"S{i}.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0,
                    atr=2.0, stop_distance=1.5, quantity=1000, position_value=1e5, turnover_cr=50.0,
                    reserve=i > 8, limit_price=99.25) for i in range(1, 11)]
    info = {"ranked_by": "ranker", "guard_t": 1.2, "drift_alarm": False}
    monkeypatch.setattr(gs, "wait_for_clock", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_publish", lambda s, fetch, history_days: (date(2026, 10, 5), date(2026, 10, 1),
                                                                        picks, info))
    from nse_intraday_ai import cloud_sync
    synced = []
    monkeypatch.setattr(cloud_sync, "publish", lambda paths, message, **kw: synced.append(message))
    gs.cmd_picks(SimpleNamespace(date="2026-10-05", push=False, no_fetch=True, history_days=20))
    assert synced == ["chore(data): publish picks for 2026-10-05 [skip ci]"]     # never the real repo
    out = capsys.readouterr().out
    assert "LIMIT ₹99.25" in out
    assert "CANCEL any unfilled order" in out
    assert "use the next: S9, S10" in out
    assert "Ranked by the trained model" in out


def test_a_list_built_the_evening_before_is_marked_preliminary(monkeypatch, capsys):
    gs = _load_gap_script()
    G = gs.G
    picks = [G.Pick(rank=i, symbol=f"S{i}.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0,
                    atr=2.0, stop_distance=1.5, quantity=1000, position_value=1e5, turnover_cr=50.0,
                    reserve=i > 8) for i in range(1, 11)]
    monkeypatch.setattr(gs, "wait_for_clock", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_publish", lambda s, fetch, history_days: (date(2099, 1, 2), date(2099, 1, 1),
                                                                        picks, {"ranked_by": "ranker"}))
    from nse_intraday_ai import cloud_sync
    monkeypatch.setattr(cloud_sync, "publish", lambda *a, **k: None)
    gs.cmd_picks(SimpleNamespace(date="2099-01-02", push=False, no_fetch=True, history_days=20))
    out = capsys.readouterr().out
    assert "Preliminary gap-reversal shorts" in out and "if no 08:15 list arrives, trade this one" in out


def _load_health():
    spec = importlib.util.spec_from_file_location("health_script", ROOT / "scripts" / "health_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_health_flags_a_session_the_learner_skipped_and_a_day_without_a_list(tmp_path, monkeypatch):
    import json
    from datetime import datetime

    hc = _load_health()
    monkeypatch.setattr(hc, "ROOT", tmp_path)
    for d in ("20261001", "20261005", "20261006"):
        f = tmp_path / "data" / "nse_bhav" / "2026" / f"bhav_{d}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    (models / "meta_state.json").write_text(json.dumps(
        {"updated_through": "2026-10-06", "track": [{"session": "2026-10-01"}, {"session": "2026-10-06"}]}))
    name, status, detail = hc.check_learning()
    assert status == hc.WARN and "2026-10-05" in detail

    book = tmp_path / "data" / "gap_reversal"
    (book / "history").mkdir(parents=True)
    (book / "paper_book.csv").write_text("session,symbol,net\n2026-10-01,A.NS,1.0\n")
    (book / "history" / "picks_20261005.json").write_text("{}")
    _, status, detail = hc.check_published(datetime(2026, 10, 6, 20, 0, tzinfo=hc.IST))
    assert status == hc.WARN and detail.startswith("no list for 2026-10-06")


def test_cloud_fallback_stands_aside_only_for_the_laptops_final_list():
    gs = _load_gap_script()
    today = date(2026, 10, 12)
    final = {"session": "2026-10-12", "generated_at": "2026-10-12T08:15:40+05:30", "publisher": "laptop"}
    assert gs.laptop_already_published(final, today)
    assert not gs.laptop_already_published({**final, "publisher": "github-actions"}, today)
    assert not gs.laptop_already_published({**final, "generated_at": "2026-10-11T20:30:00+05:30"}, today)  # preliminary
    assert not gs.laptop_already_published({**final, "session": "2026-10-09"}, today)                    # stale
    legacy = {"session": "2026-10-12", "generated_at": "2026-10-12T08:45:10+05:30", "model": {"n_train": 1}}
    assert gs.laptop_already_published(legacy, today)
    assert not gs.laptop_already_published({k: v for k, v in legacy.items() if k != "model"}, today)
    assert not gs.laptop_already_published(None, today)


def _bars_until(last_bar: int):
    import pandas as pd
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.0, "close": 99.5},
                        index=range(last_bar + 1))


def test_record_refuses_a_finished_day_whose_bars_stop_early(monkeypatch):
    # 2026-10-27: after a resume with no network the cache held bars to 14:15 only,
    # and five of eight trades were booked at 14:15 instead of 15:15.
    from datetime import datetime

    gs = _load_gap_script()
    G = gs.G
    picks = [G.Pick(rank=i, symbol=f"S{i}.NS", side="SHORT", gap_prev_pct=1.0, prev_close=100.0, atr=3.0,
                    stop_distance=2.25, quantity=1250, position_value=1.25e5, turnover_cr=50.0)
             for i in range(1, 9)]
    monkeypatch.setattr(G, "load_picks", lambda s: picks)
    monkeypatch.setattr(gs, "official_opens", lambda s: {})
    monkeypatch.setattr(G, "load_session_bars", lambda sym, s: _bars_until(60))      # 09:15 + 60*5 = 14:15
    with pytest.raises(gs.IncompleteSession):
        gs.record_session(date(2026, 10, 27), fetch=False, now=datetime(2026, 10, 27, 19, 15, tzinfo=gs.IST))
    # during the session the same bars are simply "so far"
    assert len(gs.record_session(date(2026, 10, 27), fetch=False,
                                 now=datetime(2026, 10, 27, 14, 20, tzinfo=gs.IST))) == 8
    monkeypatch.setattr(G, "load_session_bars", lambda sym, s: _bars_until(74))
    trades = gs.record_session(date(2026, 10, 27), fetch=False, now=datetime(2026, 10, 27, 19, 15, tzinfo=gs.IST))
    assert {t.exit_reason for t in trades} == {"SQUARE_OFF"}


def test_official_opens_fall_back_to_the_bhavcopy(monkeypatch):
    import pandas as pd

    gs = _load_gap_script()
    from nse_intraday_ai import nse_bhav, nse_preopen
    monkeypatch.setattr(nse_preopen, "archived_opens", lambda d: {})
    monkeypatch.setattr(nse_bhav, "load", lambda a, b: pd.DataFrame(
        {"symbol": ["A", "B"], "series": ["EQ", "BE"], "open": [101.5, 50.0]}))
    assert gs.official_opens(date(2026, 10, 27)) == {"A": 101.5}
    monkeypatch.setattr(nse_preopen, "archived_opens", lambda d: {"A": 101.0})
    assert gs.official_opens(date(2026, 10, 27)) == {"A": 101.0}


def test_record_takes_late_lists_out_of_the_paper_book(tmp_path, monkeypatch):
    import json

    import pandas as pd

    gs = _load_gap_script()
    G = gs.G
    monkeypatch.setattr(gs, "wait_for_clock", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_sync", lambda *a, **k: None)
    paper = tmp_path / "paper_book.csv"
    monkeypatch.setattr(gs, "PAPER", paper)
    monkeypatch.setattr(gs, "OUT", tmp_path)
    monkeypatch.setattr(G, "PICKS_PATH", tmp_path / "picks.json")
    hist = tmp_path / "history"
    hist.mkdir()
    for d, stamp in (("2026-10-14", "2026-10-14T11:08:27+05:30"), ("2026-10-15", "2026-10-15T08:58:31+05:30")):
        (hist / f"picks_{d.replace('-', '')}.json").write_text(json.dumps({"session": d, "generated_at": stamp, "picks": []}))
    pd.DataFrame({"session": ["2026-10-14", "2026-10-15"], "symbol": ["A.NS", "B.NS"], "net": [-100.0, 50.0]}
                 ).to_csv(paper, index=False)
    gs.cmd_record(SimpleNamespace(date="2026-10-15", no_fetch=True, push=False))
    assert list(pd.read_csv(paper)["session"]) == ["2026-10-15"]


def test_a_morning_retrain_waits_for_the_days_list(monkeypatch):
    from datetime import datetime

    spec = importlib.util.spec_from_file_location("learn_script", ROOT / "scripts" / "learn.py")
    learn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learn)
    from nse_intraday_ai import gap_reversal as G
    from nse_intraday_ai import nse_calendar

    monkeypatch.setattr(nse_calendar, "is_trading_day", lambda d: True)
    t = [datetime(2026, 10, 28, 8, 5)]
    payloads = [None, None, {"session": "2026-10-28", "published_at": "2026-10-28T08:16:02"}]
    monkeypatch.setattr(G, "read_picks", lambda *a: payloads.pop(0))

    def tick(seconds):
        t[0] = t[0].replace(minute=t[0].minute + 5)
    assert learn.wait_for_todays_list(lambda: t[0], sleep=tick) is True
    assert t[0].strftime("%H:%M") == "08:15"
    # in the evening it does not wait at all
    assert learn.wait_for_todays_list(lambda: datetime(2026, 10, 28, 19, 0), sleep=tick) is False


def _load_keep_awake():
    spec = importlib.util.spec_from_file_location("keep_awake", ROOT / "scripts" / "keep_awake.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_keep_awake_lets_go_when_unplugged_or_when_the_window_ends(tmp_path):
    from datetime import datetime, timedelta

    ka = _load_keep_awake()
    t = [datetime(2026, 10, 28, 9, 0)]

    def tick(seconds):
        t[0] += timedelta(seconds=seconds)
    end = datetime(2026, 10, 28, 16, 15)
    plugged = iter([True] * 10 + [False])
    assert ka.hold(end, clock=lambda: t[0], sleep=tick, mains=lambda: next(plugged)) == "unplugged"
    assert t[0] < end
    t[0] = datetime(2026, 10, 28, 16, 0)
    assert ka.hold(end, clock=lambda: t[0], sleep=tick, mains=lambda: True) == "window over"
    assert ka.until_today("1615", datetime(2026, 10, 28, 8, 0)) == end
    (tmp_path / "ACAD").mkdir()
    (tmp_path / "ACAD" / "online").write_text("0\n")
    (tmp_path / "BAT0").mkdir()
    assert ka.on_mains(tmp_path) is False
    (tmp_path / "ACAD" / "online").write_text("1\n")
    assert ka.on_mains(tmp_path) is True


def test_the_evening_lock_runs_through_the_night_only_before_a_session():
    from datetime import date, datetime

    ka = _load_keep_awake()
    trading = lambda d: d.weekday() < 5 and d != date(2026, 11, 9)          # noqa: E731
    sun = datetime(2026, 11, 1, 20, 50)
    assert ka.window_end("2110", sun, next_session="1615", is_trading_day=trading) == datetime(2026, 11, 2, 16, 15)
    fri = datetime(2026, 10, 30, 20, 50)                                     # Saturday is not a session
    assert ka.window_end("2110", fri, next_session="1615", is_trading_day=trading) == datetime(2026, 10, 30, 21, 10)
    holiday_eve = datetime(2026, 11, 8, 20, 50)
    assert ka.window_end("2110", holiday_eve, next_session="1615", is_trading_day=trading).hour == 21
    assert ka.window_end("1615", datetime(2026, 10, 28, 7, 55)) == datetime(2026, 10, 28, 16, 15)


def test_health_repushes_when_the_cloud_copy_is_behind(tmp_path, monkeypatch):
    import json

    hc = _load_health()
    monkeypatch.setattr(hc, "ROOT", tmp_path)
    picks = tmp_path / "data" / "gap_reversal" / "picks.json"
    picks.parent.mkdir(parents=True)
    picks.write_text(json.dumps({"session": "2026-10-28", "generated_at": "2026-10-28T08:15:40+05:30"}))
    from nse_intraday_ai import cloud_sync
    main = {"session": "2026-10-28", "generated_at": "2026-10-27T21:15:10+05:30"}     # last evening's
    monkeypatch.setattr(cloud_sync, "read_remote", lambda path, **kw: json.dumps(main))
    pushed = []

    def publish(paths, message, **kw):
        pushed.append(message)
        main.update(json.loads(picks.read_text()))
        return True
    monkeypatch.setattr(cloud_sync, "publish", publish)
    name, status, detail = hc.check_cloud_copy()
    assert status == hc.OK and pushed and "re-pushed" in detail
    pushed.clear()
    assert hc.check_cloud_copy()[1] == hc.OK and not pushed                 # in sync: nothing to do


def test_time_bound_jobs_ignore_a_boot_time_burst():
    # 2026-10-28 07:54: the clock jumped from "June 5" at boot, every timer fired,
    # and the 09:18 job pushed "No opening prices ... No trades today".
    from datetime import datetime

    gs = _load_gap_script()
    at = lambda hm: datetime(2026, 10, 28, *map(int, hm.split(":")), tzinfo=gs.IST)   # noqa: E731
    assert gs.outside_window("09:15", "15:30", at("07:54"))
    assert not gs.outside_window("09:15", "15:30", at("09:18"))
    assert gs.outside_window("09:05", "09:20", at("07:54")) and not gs.outside_window("09:05", "09:20", at("09:10"))
    ka = _load_keep_awake()
    assert ka.evening_anchor("2045", datetime(2026, 10, 28, 7, 54)) is None            # daytime: no lock
    assert ka.evening_anchor("2045", datetime(2026, 10, 28, 21, 0)) == datetime(2026, 10, 28, 21, 0)
    assert ka.evening_anchor("2045", datetime(2026, 10, 29, 2, 0)) == datetime(2026, 10, 28, 2, 0)


def test_levels_before_the_open_sends_nothing(monkeypatch, capsys):
    gs = _load_gap_script()
    monkeypatch.setattr(gs, "wait_for_clock", lambda *a, **k: None)
    monkeypatch.setattr(gs, "outside_window", lambda *a, **k: True)
    sent = []
    monkeypatch.setattr(gs, "push", lambda *a, **k: sent.append(a))
    gs.cmd_levels(SimpleNamespace(date=None, push=True, no_fetch=True))
    assert sent == [] and "no stop levels before the 09:15 open" in capsys.readouterr().out
