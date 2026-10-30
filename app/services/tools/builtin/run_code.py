"""Run-code tool: execute Python or shell code in a sandboxed subprocess.

The tool runs code in a temporary directory with a timeout. Requires
confirmation before executing, since code can do anything.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.services.tools.registry import Tool, register

_TIMEOUT_SECONDS = 30


def run_code(language: str, code: str) -> dict[str, Any]:
    """Execute code in the given language and return stdout, stderr, and return code."""
    language = language.lower()
    if language not in ("python", "shell", "sh", "bash"):
        raise ValueError(f"Unsupported language: {language}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if language == "python":
            script = tmp_path / "script.py"
            script.write_text(code, encoding="utf-8")
            cmd = [sys.executable, str(script)]
        else:
            script = tmp_path / "script.sh"
            script.write_text(code, encoding="utf-8")
            script.chmod(0o755)
            cmd = ["/bin/zsh", str(script)]
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=tmpdir,
            check=False,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "language": language,
        }


register(
    Tool(
        name="run_code",
        description="Execute Python or shell code. Returns stdout, stderr, and return code.",
        parameters={
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "shell"],
                    "description": "The code language: python or shell.",
                },
                "code": {
                    "type": "string",
                    "description": "The code to execute.",
                },
            },
            "required": ["language", "code"],
        },
        requires_confirmation=True,
        handler=run_code,
    )
)
