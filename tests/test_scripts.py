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
    assert "Preliminary gap-reversal shorts" in out and "The 08:15 list replaces it" in out


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
