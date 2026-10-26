"""Start the Luke backend from one command.

The launcher runs as `start engine` (the launcher name and the command word are
free-form); this module owns the checks it runs before uvicorn takes over: the
virtual environment, requirements, the spaCy model, Postgres and the uvicorn
process itself.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"
DEFAULT_DATABASE_PORT = 5433
SERVER_PORT = 8000
DOCKER_APP = Path("/Applications/Docker.app")
DOCKER_PATHS = (
    Path.home() / ".docker" / "bin" / "docker",
    Path("/usr/local/bin/docker"),
    Path("/opt/homebrew/bin/docker"),
    DOCKER_APP / "Contents" / "Resources" / "bin" / "docker",
)
DOCKER_WAIT_SECONDS = 90
DATABASE_WAIT_SECONDS = 30


class StartupError(RuntimeError):
    """A startup step that needs the user's attention."""


def _say(message: str) -> None:
    """Print a startup line now; execve replaces the process without flushing."""
    print(message, flush=True)


def _run(
    command: list[str], *, capture: bool = False, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a fixed command from the backend folder."""
    try:
        return subprocess.run(  # noqa: S603 -- fixed argv list, never a shell
            command,
            cwd=ROOT,
            check=False,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(command, 1, "", str(error))


def _setting(key: str, default: str) -> str:
    """An environment value, then the .env file, then the default."""
    if key in os.environ:
        return os.environ[key]
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return default
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("\"'")
    return default


def _database_port(url: str | None = None) -> int:
    """The port the configured database listens on."""
    address = url or _setting(
        "DATABASE_URL",
        f"postgresql+psycopg2://postgres:postgres@localhost:{DEFAULT_DATABASE_PORT}/assistance",
    )
    try:
        port = urlsplit(address).port
    except ValueError:
        return DEFAULT_DATABASE_PORT
    return port or DEFAULT_DATABASE_PORT


def _requirements_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requirements_stale(requirements: Path = REQUIREMENTS, stamp: Path = STAMP) -> bool:
    """Whether the environment was installed from a different requirements file."""
    if not requirements.is_file():
        raise StartupError(f"{requirements.name} is missing.")
    if not stamp.is_file():
        return True
    return stamp.read_text(encoding="utf-8").strip() != _requirements_digest(requirements)


def _ensure_venv() -> None:
    if PYTHON.is_file() and os.access(PYTHON, os.X_OK):
        _say(f"Using {VENV.name}.")
        return
    if VENV.exists():
        raise StartupError(f"{VENV} has no working Python. Remove it and run this again.")
    _say(f"Creating {VENV.name}…")
    result = _run([sys.executable, "-m", "venv", str(VENV)])
    if result.returncode != 0 or not PYTHON.is_file():
        raise StartupError(f"Could not create {VENV.name}.")
    _say(f"Using {VENV.name}.")


def _install_requirements() -> None:
    if not _requirements_stale():
        _say("Python requirements are up to date.")
        return
    _say("Installing requirements.txt…")
    result = _run([str(PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if result.returncode != 0:
        raise StartupError("pip could not install requirements.txt.")
    STAMP.write_text(_requirements_digest(REQUIREMENTS), encoding="utf-8")
    _say("Python requirements installed.")


def _ensure_spacy() -> None:
    model = _setting("SPACY_MODEL", "en_core_web_sm")
    _say(f"Checking the spaCy model {model}…")
    probe = [str(PYTHON), "-c", "import spacy, sys; spacy.load(sys.argv[1])", model]
    if _run(probe, capture=True).returncode == 0:
        return
    _say(f"Downloading the spaCy model {model}…")
    result = _run([str(PYTHON), "-m", "spacy", "download", model])
    if result.returncode != 0:
        raise StartupError(f"Could not download the spaCy model {model}.")


def _docker_path() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    for candidate in DOCKER_PATHS:
        if candidate.is_file():
            return str(candidate)
    return None


def _docker_running(docker: str) -> bool:
    return _run([docker, "info"], capture=True, timeout=20).returncode == 0


def _start_docker(docker: str) -> None:
    if _docker_running(docker):
        return
    if sys.platform != "darwin" or not DOCKER_APP.is_dir():
        raise StartupError("Docker is not running. Start it and run `start engine` again.")
    _say("Docker is not running. Opening Docker Desktop…")
    _run(["open", "-a", "Docker"])
    deadline = time.monotonic() + DOCKER_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _docker_running(docker):
            _say("Docker is ready.")
            return
        time.sleep(2)
    raise StartupError(f"Docker did not start within {DOCKER_WAIT_SECONDS} seconds.")


def _compose_up(docker: str) -> None:
    _say("Starting Postgres with Docker Compose…")
    result = _run([docker, "compose", "up", "-d", "postgres"])
    if result.returncode != 0:
        raise StartupError("Docker Compose could not start Postgres.")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _wait_for_database(port: int) -> bool:
    deadline = time.monotonic() + DATABASE_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(1)
    return _port_open(port)


def _ensure_postgres() -> None:
    port = _database_port()
    if _port_open(port):
        _say(f"Postgres is already listening on port {port}.")
        return
    docker = _docker_path()
    if docker is None:
        raise StartupError(
            f"Postgres is not listening on port {port}, and Docker was not found. "
            "Start Postgres and run `start engine` again."
        )
    _start_docker(docker)
    _compose_up(docker)
    if not _wait_for_database(port):
        raise StartupError(
            f"Postgres did not open port {port}. Check `docker compose logs postgres`."
        )
    _say(f"Postgres is ready on port {port}.")


def _uvicorn_command() -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
    ]


def _server_environment() -> dict[str, str]:
    """The environment a terminal would have after activating the venv."""
    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(VENV)
    environment["PATH"] = f"{VENV / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    return environment


def _start_engine() -> None:
    os.chdir(ROOT)
    _say("Starting the Luke engine.")
    _ensure_venv()
    _install_requirements()
    _ensure_spacy()
    _ensure_postgres()
    command = _uvicorn_command()
    _say(f"Uvicorn is starting on http://localhost:{SERVER_PORT}. Press Ctrl+C to stop.")
    try:
        os.execve(  # noqa: S606 -- fixed argv, no shell wanted
            command[0], command, _server_environment()
        )
    except OSError as error:
        raise StartupError(f"Could not start uvicorn: {error}") from error


def main() -> int:
    try:
        _start_engine()
    except StartupError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
