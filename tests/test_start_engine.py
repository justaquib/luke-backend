"""The start script's decisions, without starting Docker or the server."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import start_engine

LAUNCHER = Path(__file__).resolve().parents[1] / "start"


def test_the_database_port_is_read_from_the_configured_url():
    assert (
        start_engine._database_port(
            "postgresql+psycopg2://postgres:postgres@localhost:5433/assistance"
        )
        == 5433
    )
    assert (
        start_engine._database_port("postgresql+psycopg2://postgres:postgres@localhost/assistance")
        == 5433
    )


def test_requirements_are_installed_only_while_the_file_is_new(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi==1.0\n", encoding="utf-8")
    stamp = tmp_path / "requirements.sha256"

    assert start_engine._requirements_stale(requirements, stamp) is True

    stamp.write_text(start_engine._requirements_digest(requirements), encoding="utf-8")
    assert start_engine._requirements_stale(requirements, stamp) is False

    requirements.write_text("fastapi==2.0\n", encoding="utf-8")
    assert start_engine._requirements_stale(requirements, stamp) is True


def test_the_server_runs_from_the_virtual_environment_on_port_8000():
    command = start_engine._uvicorn_command()

    assert command == [
        str(start_engine.PYTHON),
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_a_database_that_is_already_listening_does_not_need_docker(monkeypatch):
    monkeypatch.setattr(start_engine, "_database_port", lambda: 5433)
    monkeypatch.setattr(start_engine, "_port_open", lambda port: True)
    monkeypatch.setattr(
        start_engine, "_docker_path", lambda: pytest.fail("Docker should not be started")
    )

    start_engine._ensure_postgres()


def test_a_closed_database_without_docker_is_reported(monkeypatch):
    monkeypatch.setattr(start_engine, "_database_port", lambda: 5433)
    monkeypatch.setattr(start_engine, "_port_open", lambda port: False)
    monkeypatch.setattr(start_engine, "_docker_path", lambda: None)

    with pytest.raises(start_engine.StartupError, match="Docker was not found"):
        start_engine._ensure_postgres()


def test_a_closed_database_starts_docker_and_waits_for_the_port(monkeypatch):
    calls = []
    monkeypatch.setattr(start_engine, "_database_port", lambda: 5433)
    monkeypatch.setattr(start_engine, "_port_open", lambda port: False)
    monkeypatch.setattr(start_engine, "_docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(start_engine, "_start_docker", lambda docker: calls.append("docker"))
    monkeypatch.setattr(start_engine, "_compose_up", lambda docker: calls.append("compose"))
    monkeypatch.setattr(start_engine, "_wait_for_database", lambda port: True)

    start_engine._ensure_postgres()

    assert calls == ["docker", "compose"]


def _fake_python(tmp_path: Path) -> Path:
    """A python3 that reports the script it was asked to run."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.write_text('#!/usr/bin/env bash\necho "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    return fake_bin


def test_a_renamed_launcher_runs_the_engine_for_any_command_word(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    engine = scripts / "start_engine.py"
    engine.write_text("", encoding="utf-8")
    launcher = tmp_path / "launcher"
    shutil.copy(LAUNCHER, launcher)
    launcher.chmod(0o755)
    fake_bin = _fake_python(tmp_path)
    environment = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(  # noqa: S603 -- fixed launcher copy, no shell
        [str(launcher), "luke"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == str(engine)


def test_the_launcher_usage_uses_its_own_name():
    result = subprocess.run(  # noqa: S603 -- fixed launcher path, no shell
        [str(LAUNCHER)], check=False, capture_output=True, text=True
    )

    assert result.returncode == 2
    assert f"usage: {LAUNCHER.name} <command>" in result.stderr


def test_the_spacy_check_is_announced_before_it_runs(monkeypatch, capsys):
    monkeypatch.setattr(start_engine, "_setting", lambda key, default: "en_core_web_sm")
    monkeypatch.setattr(
        start_engine,
        "_run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    start_engine._ensure_spacy()

    assert "Checking the spaCy model en_core_web_sm…" in capsys.readouterr().out


def _interrupt():
    raise KeyboardInterrupt


def test_ctrl_c_stops_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(start_engine, "_start_engine", _interrupt)

    assert start_engine.main() == 130
    assert "Stopped." in capsys.readouterr().err


def test_progress_is_written_before_uvicorn_replaces_the_process():
    program = "\n".join(
        (
            "import os, sys",
            "from scripts.start_engine import _say",
            "_say('starting')",
            "os.execve(sys.executable, [sys.executable, '-c', \"print('serving')\"], os.environ)",
        )
    )

    result = subprocess.run(  # noqa: S603 -- fixed interpreter, no shell
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["starting", "serving"]
