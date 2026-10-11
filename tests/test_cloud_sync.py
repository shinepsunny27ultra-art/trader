from __future__ import annotations

import subprocess

from nse_intraday_ai import cloud_sync


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def test_publish_pushes_to_main_without_touching_the_checked_out_branch(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(remote))
    _git(tmp_path, "init", "--quiet", "-b", "main", str(work))
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git(work, "config", k, v)
    (work / "app.py").write_text("v1\n")
    _git(work, "add", "app.py")
    _git(work, "commit", "--quiet", "-m", "init")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "--quiet", "origin", "main")
    _git(work, "checkout", "--quiet", "-b", "devel")
    (work / "staged.py").write_text("work in progress\n")
    _git(work, "add", "staged.py")                                  # someone's staged edit
    data = work / "data" / "picks.json"
    data.parent.mkdir()
    data.write_text('{"session": "2026-10-07"}\n')
    devel_head = _git(work, "rev-parse", "HEAD")

    assert cloud_sync.publish([data], "chore(data): picks", root=work, log=lambda *a: None)
    assert _git(remote, "show", "main:data/picks.json") == '{"session": "2026-10-07"}'
    assert _git(remote, "show", "main:app.py") == "v1"
    assert _git(work, "rev-parse", "HEAD") == devel_head              # branch untouched
    assert _git(work, "diff", "--cached", "--name-only") == "staged.py"   # index untouched
    pushed = _git(remote, "rev-parse", "main")
    assert cloud_sync.publish([data], "again", root=work, log=lambda *a: None)
    assert _git(remote, "rev-parse", "main") == pushed               # no empty commits


def test_publish_is_a_no_op_inside_github_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    (tmp_path / "f").write_text("x")
    assert cloud_sync.publish([tmp_path / "f"], "m", root=tmp_path) is False
