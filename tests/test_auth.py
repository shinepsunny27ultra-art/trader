"""Dashboard password gate: hashed file or DASHBOARD_PASSWORD secret, never a built-in default."""
from __future__ import annotations

from nse_intraday_ai import auth


def test_fails_closed_without_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    path = tmp_path / "auth.json"
    assert not auth.is_configured(path)
    for guess in ("", "password", "0480336699", "123456"):
        assert not auth.verify(guess, path)


def test_hashed_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    path = tmp_path / "auth.json"
    auth.set_password("correct-horse", path)
    assert auth.is_configured(path)
    assert auth.verify("correct-horse", path) and not auth.verify("wrong-horse", path)
    assert "correct-horse" not in path.read_text()


def test_env_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "cloud-secret-1")
    path = tmp_path / "auth.json"
    assert auth.is_configured(path)
    assert auth.verify("cloud-secret-1", path) and not auth.verify("cloud-secret-2", path)
