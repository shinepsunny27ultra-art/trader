"""Publish data files to GitHub's `main` for the Streamlit Cloud copy of the app.

The commit is built with git plumbing in a throwaway index on top of the remote
branch and pushed from there, so the checked-out branch, its index and the
working tree are never touched.  (The first version ran `git add/commit` on
whatever branch was checked out — `devel` collected the data commits — and then
pushed the unrelated local `main`, so nothing reached the cloud.)

Skipped inside GitHub Actions: the workflow commits its own output.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def state_files(root: Path = ROOT) -> list[Path]:
    """Small files the cloud dashboard reads besides the live scan: the list, the
    paper record, the backtest summary and the model/monitoring summaries.  Model
    binaries and raw market data stay on the laptop (the repo is public)."""
    data = root / "data"
    return [data / "gap_reversal" / "picks.json", data / "gap_reversal" / "paper_book.csv",
            data / "gap_reversal" / "backtest.json", data / "models" / "champion.json",
            data / "models" / "weekly_report.json", data / "models" / "meta_state.json"]


def read_remote(path: Path, *, branch: str = "main", remote: str = "origin", root: Path = ROOT,
                timeout: float = 30) -> str | None:
    """Text of `path` as it is on `remote/branch` right now; None if unavailable."""
    try:
        rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,  # noqa: E731
                                        timeout=timeout, check=True).stdout
        run("fetch", "--quiet", remote, branch)
        return run("show", f"FETCH_HEAD:{rel}")
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def publish(paths: list[Path], message: str, *, branch: str = "main", remote: str = "origin",
            root: Path = ROOT, timeout: float = 30, attempts: int = 4, log=print,
            force: bool = False, guard=None, backoff: float = 10.0) -> bool:
    """Commit `paths` (as they are on disk) onto `remote/branch` and push. True on success.

    Inside GitHub Actions this is a no-op unless `force` (the cloud fallback).
    `guard(read)` is called with a reader of the files on the commit the new
    commit will be built on (`read(rel_path) -> str | None`); if it returns
    False nothing is pushed.  A push only succeeds if `branch` still points at
    that commit, so the guard's verdict cannot go stale.
    """
    if os.environ.get("GITHUB_ACTIONS") and not force:
        return False
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        return False

    def git(*args: str, env: dict | None = None) -> str:
        return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True,
                              timeout=timeout, check=True).stdout.strip()

    for attempt in range(attempts):
        try:
            git("fetch", "--quiet", remote, branch)
            base = git("rev-parse", "FETCH_HEAD")
            if guard is not None:
                def read(rel: str, base: str = base) -> str | None:
                    try:
                        return git("show", f"{base}:{rel}")
                    except subprocess.SubprocessError:
                        return None
                if not guard(read):
                    log("  cloud sync: guard refused (main changed); nothing pushed")
                    return False
            with tempfile.TemporaryDirectory() as tmp:
                env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
                git("read-tree", base, env=env)
                for path in paths:
                    rel = path.resolve().relative_to(root.resolve()).as_posix()
                    blob = git("hash-object", "-w", "--", str(path))
                    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
                tree = git("write-tree", env=env)
            if tree == git("rev-parse", f"{base}^{{tree}}"):
                return True                                   # already up to date
            commit = git("commit-tree", tree, "-p", base, "-m", message)
            git("push", "--quiet", remote, f"{commit}:refs/heads/{branch}")
            return True
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            # A push can lose a race with the GitHub Action, or the hotspot can drop
            # for a moment: refetch and retry after a pause (10 s, 20 s, 30 s).
            detail = getattr(exc, "stderr", "") or exc
            if attempt == attempts - 1:
                log(f"  cloud sync failed: {str(detail).strip()[:200]}")
            else:
                import time
                time.sleep(backoff * (attempt + 1))
    return False
