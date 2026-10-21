"""Hard structure rules that Ruff does not cover.

Run with: python scripts/check_structure.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_FILE_LINES = 400
MAX_FUNCTION_LINES = 60
MAX_CLASS_LINES = 250
MAX_MAIN_LINES = 120
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
TESTS = ROOT / "tests"
IGNORED_PARTS = {
    ".venv",
    "__pycache__",
    ".chroma",
    ".language_model",
    "uploads",
    "legacy",
}
BANNED_IMPORTS = {"numpy", "scikit-learn", "transformers"}


def python_files() -> list[Path]:
    files: list[Path] = []
    for directory in (APP, TESTS):
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*.py")
            if not any(part in IGNORED_PARTS for part in path.parts)
        )
    return sorted(files)


def node_lines(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    start = getattr(node, "lineno", None)
    if end is None or start is None:
        return 0
    return end - start + 1


def route_handlers(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    handlers: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = ""
            if isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            if name in {"get", "post", "put", "patch", "delete"}:
                handlers.append(node)
                break
    return handlers


def imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def silent_excepts(tree: ast.AST) -> list[ast.ExceptHandler]:
    found: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
            continue
        found.append(node)
    return found


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    relative = path.relative_to(ROOT)
    if len(lines) > MAX_FILE_LINES:
        problems.append(f"{relative}:1 file has {len(lines)} lines (max {MAX_FILE_LINES})")

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"{relative}:{error.lineno} syntax error: {error.msg}"]

    if relative == Path("app/main.py") and len(lines) > MAX_MAIN_LINES:
        problems.append(f"{relative}:1 app/main.py has {len(lines)} lines (max {MAX_MAIN_LINES})")
    if relative == Path("app/main.py"):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if isinstance(target, ast.Attribute) and target.attr in {
                        "get",
                        "post",
                        "put",
                        "patch",
                        "delete",
                    }:
                        problems.append(
                            f"{relative}:{node.lineno} route '{node.name}' belongs in app/routers"
                        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node_lines(node)
            if length > MAX_FUNCTION_LINES:
                problems.append(
                    f"{relative}:{node.lineno} '{node.name}' has {length} lines "
                    f"(max {MAX_FUNCTION_LINES})"
                )
        elif isinstance(node, ast.ClassDef):
            length = node_lines(node)
            if length > MAX_CLASS_LINES:
                problems.append(
                    f"{relative}:{node.lineno} class '{node.name}' has {length} lines "
                    f"(max {MAX_CLASS_LINES})"
                )

    for handler in route_handlers(tree):
        length = node_lines(handler)
        if length > MAX_FUNCTION_LINES:
            problems.append(
                f"{relative}:{handler.lineno} route '{handler.name}' has {length} lines "
                f"(max {MAX_FUNCTION_LINES})"
            )

    for module in imported_modules(tree):
        if module in BANNED_IMPORTS:
            problems.append(f"{relative}:1 banned import '{module}'")

    for handler in silent_excepts(tree):
        problems.append(f"{relative}:{handler.lineno} silent exception handler")

    if relative.parts[0] == "app" and "print(" in source:
        problems.append(f"{relative}:1 use logging, not print()")

    if relative.parts[0] == "app":
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec"}
            ):
                problems.append(f"{relative}:{node.lineno} eval/exec is forbidden")

    if relative.parts[:2] == ("app", "services"):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("app.routers", "app.main")):
                    problems.append(
                        f"{relative}:{node.lineno} services must not import routers or main"
                    )

    if "TODO" in source or "FIXME" in source:
        problems.append(f"{relative}:1 remove TODO/FIXME or link an issue")

    return problems


def main() -> int:
    problems: list[str] = []
    files = python_files()
    for path in files:
        problems.extend(check_file(path))
    if problems:
        print("Structure rules failed:\n")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"Checked {len(files)} Python files: structure OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
