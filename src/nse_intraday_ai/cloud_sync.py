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


def publish(paths: list[Path], message: str, *, branch: str = "main", remote: str = "origin",
            root: Path = ROOT, timeout: float = 30, attempts: int = 2, log=print) -> bool:
    """Commit `paths` (as they are on disk) onto `remote/branch` and push. True on success."""
    if os.environ.get("GITHUB_ACTIONS"):
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
            # A push can lose a race with the GitHub Action; refetch and retry once.
            detail = getattr(exc, "stderr", "") or exc
            if attempt == attempts - 1:
                log(f"  cloud sync failed: {str(detail).strip()[:200]}")
    return False
