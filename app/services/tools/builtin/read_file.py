"""Read-file tool: read the contents of a file.

The tool reads any file the OS user can read. No sandboxing: the user has
already confirmed they trust Luke with filesystem access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.tools.registry import Tool, register


def read_file(path: str) -> dict[str, Any]:
    """Read the contents of a file and return them as text."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")
    return {
        "path": str(file_path),
        "content": content,
        "size_bytes": file_path.stat().st_size,
    }


register(
    Tool(
        name="read_file",
        description="Read the contents of a file. Returns the text and file size.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path to the file.",
                }
            },
            "required": ["path"],
        },
        requires_confirmation=False,
        handler=read_file,
    )
)
