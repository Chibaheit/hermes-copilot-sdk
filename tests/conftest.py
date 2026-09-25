"""External-suite isolation: establish a disposable home before importing Hermes."""

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "tests" / ".sandbox" / uuid4().hex
SANDBOX.mkdir(parents=True)
for key in tuple(os.environ):
    if any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
        os.environ.pop(key)
for key in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "TMPDIR"):
    os.environ[key] = str(SANDBOX)
os.environ["HERMES_HOME"] = str(SANDBOX / "launch")
os.environ["HERMES_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(ROOT.parent / "hermes-agent"))
atexit.register(shutil.rmtree, SANDBOX, ignore_errors=True)


def pytest_addoption(parser):
    parser.addoption(
        "--run-copilot-runtime-smoke", action="store_true", default=False,
        help="Run the cached Copilot runtime protocol test (no auth or inference)",
    )


def pytest_configure(config):
    config.option.basetemp = str(SANDBOX / "pytest")


@dataclass(frozen=True)
class Profile:
    home: Path
    token: str
    timeout: float

    @contextmanager
    def activate(self):
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_secret_scope,
        )
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home_token = set_hermes_home_override(self.home)
        secret_token = set_secret_scope(build_profile_secret_scope(self.home))
        try:
            yield self
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    return home


@pytest.fixture
def profiles(tmp_path):
    result = []
    for name, timeout in (("A", 11.0), ("B", 17.0)):
        home = tmp_path / name
        home.mkdir()
        token = f"synthetic-test-token-{name}"
        (home / ".env").write_text(
            f"COPILOT_GITHUB_TOKEN={token}\nGITHUB_TOKEN=unused-{name}\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "copilot-sdk", "model": f"model-{name}"},
            "copilot_sdk": {"timeout_seconds": timeout},
            "plugins": {"enabled": ["copilot-sdk"]},
        }), encoding="utf-8")
        result.append(Profile(home, token, timeout))
    return tuple(result)
