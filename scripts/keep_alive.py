#!/usr/bin/env python3
"""Ping Streamlit Cloud and local dashboard every 10 minutes to prevent sleep.

Usage:
    python scripts/keep_alive.py                   # Pings configured URLs
    python scripts/keep_alive.py --set-url <url>   # Saves your Streamlit Cloud URL
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
URL_FILE = ROOT / "data" / "streamlit_cloud_url.txt"
LOG_FILE = ROOT / "data" / "keep_alive.log"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_urls() -> list[str]:
    urls = []
    if URL_FILE.exists():
        raw = URL_FILE.read_text().strip()
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    # Always check local dashboard as well
    urls.append("http://127.0.0.1:8501")
    return list(dict.fromkeys(urls))  # deduplicate preserving order


def ping(url: str, timeout: int = 15) -> tuple[int | None, float]:
    t0 = time.time()
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        return r.status_code, round(time.time() - t0, 3)
    except Exception as exc:  # noqa: BLE001
        return None, round(time.time() - t0, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-url", help="Save your Streamlit Cloud URL for pinging")
    ap.add_argument("--quiet", action="store_true", help="Minimal stdout")
    args = ap.parse_args()

    if args.set_url:
        URL_FILE.parent.mkdir(parents=True, exist_ok=True)
        url = args.set_url.strip()
        if not url.startswith("http"):
            url = f"https://{url}"
        URL_FILE.write_text(f"{url}\n")
        print(f"Streamlit Cloud URL saved: {url}")
        return

    now = datetime.now(IST)
    urls = get_urls()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as f:
        for url in urls:
            status, duration = ping(url)
            log_line = f"[{now:%Y-%m-%d %H:%M:%S IST}] {url} -> status={status} ({duration}s)"
            f.write(f"{log_line}\n")
            if not args.quiet:
                print(log_line)


if __name__ == "__main__":
    main()
