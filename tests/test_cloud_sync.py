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


def test_state_files_are_small_dashboard_inputs_only(tmp_path):
    files = cloud_sync.state_files(tmp_path)
    rel = {f.relative_to(tmp_path).as_posix() for f in files}
    assert "data/gap_reversal/picks.json" in rel and "data/gap_reversal/paper_book.csv" in rel
    assert "data/models/meta_state.json" in rel
    # The repo is public: no model binaries, credentials or raw market data.
    assert all(f.suffix in {".json", ".csv"} for f in files)
    assert not any("auth" in r or r.endswith(".pkl") or "candles" in r for r in rel)


def _repo_pair(tmp_path):
    """A bare 'GitHub' remote with main, a laptop clone and a runner clone."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(remote))
    clones = []
    for name in ("laptop", "runner"):
        work = tmp_path / name
        _git(tmp_path, "init", "--quiet", "-b", "main", str(work))
        for k, v in (("user.name", "t"), ("user.email", "t@t")):
            _git(work, "config", k, v)
        _git(work, "remote", "add", "origin", str(remote))
        (work / "data").mkdir()
        clones.append(work)
    laptop, runner = clones
    (laptop / "app.py").write_text("v1\n")
    _git(laptop, "add", "app.py")
    _git(laptop, "commit", "--quiet", "-m", "init")
    _git(laptop, "push", "--quiet", "origin", "main")
    _git(runner, "fetch", "--quiet", "origin")
    _git(runner, "checkout", "--quiet", "-B", "main", "origin/main")
    return remote, laptop, runner


def test_read_remote_sees_main_as_pushed(tmp_path):
    remote, laptop, runner = _repo_pair(tmp_path)
    picks = laptop / "data" / "picks.json"
    picks.write_text('{"publisher": "laptop"}\n')
    assert cloud_sync.read_remote(runner / "data" / "picks.json", root=runner) is None     # not there yet
    assert cloud_sync.publish([picks], "laptop list", root=laptop, log=lambda *a: None)
    assert cloud_sync.read_remote(runner / "data" / "picks.json", root=runner).strip() == '{"publisher": "laptop"}'


def test_fallback_guard_refuses_when_the_laptop_pushed_first(tmp_path, monkeypatch):
    remote, laptop, runner = _repo_pair(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    (runner / "data" / "picks.json").write_text('{"publisher": "github-actions"}\n')
    # The laptop's list lands between the fallback's last poll and its push.
    monkeypatch.delenv("GITHUB_ACTIONS")
    (laptop / "data" / "picks.json").write_text('{"publisher": "laptop"}\n')
    assert cloud_sync.publish([laptop / "data" / "picks.json"], "laptop", root=laptop, log=lambda *a: None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    seen = []

    def nobody_published(read):
        seen.append(read("data/picks.json"))
        return "laptop" not in (seen[-1] or "")

    ok = cloud_sync.publish([runner / "data" / "picks.json"], "fallback", root=runner, force=True,
                            guard=nobody_published, log=lambda *a: None)
    assert ok is False and seen and "laptop" in seen[0]
    assert "laptop" in _git(remote, "show", "main:data/picks.json")             # laptop list intact


def test_fallback_publishes_from_actions_only_with_force(tmp_path, monkeypatch):
    remote, laptop, runner = _repo_pair(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    f = runner / "data" / "picks.json"
    f.write_text('{"publisher": "github-actions"}\n')
    assert cloud_sync.publish([f], "no force", root=runner, log=lambda *a: None) is False
    assert cloud_sync.publish([f], "fallback", root=runner, force=True, guard=lambda read: True,
                              log=lambda *a: None)
    assert "github-actions" in _git(remote, "show", "main:data/picks.json")
    assert _git(remote, "show", "main:app.py") == "v1"                           # rest of main kept
