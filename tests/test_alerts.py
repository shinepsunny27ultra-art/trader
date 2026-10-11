from __future__ import annotations

import urllib.error

from nse_intraday_ai import alerts as A


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_ntfy_retries_while_the_network_comes_up(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise urllib.error.URLError("[Errno -3] Temporary failure in name resolution")
        return _Resp()
    monkeypatch.setattr(A.urllib.request, "urlopen", urlopen)
    assert A.send_ntfy("hi", topic="t", retry_delays=(0.0, 0.0))
    assert len(calls) == 3


def test_send_ntfy_gives_up_without_raising(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(A.urllib.request, "urlopen", urlopen)
    assert A.send_ntfy("hi", topic="t", retry_delays=(0.0,)) is False
