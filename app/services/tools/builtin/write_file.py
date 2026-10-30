"""Write-file tool: write text to a file.

The tool writes to any path the OS user can write to. Requires confirmation
before executing, since writes are destructive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.tools.registry import Tool, register


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write text to a file, creating it if it doesn't exist."""
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {
        "path": str(file_path),
        "bytes_written": len(content.encode("utf-8")),
        "created": not file_path.exists(),
    }


register(
    Tool(
        name="write_file",
        description="Write text to a file. Creates the file if it doesn't exist.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "The text to write.",
                },
            },
            "required": ["path", "content"],
        },
        requires_confirmation=True,
        handler=write_file,
    )
)
