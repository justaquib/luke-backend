"""Read the learner state files of an extracted package without trusting them."""

from __future__ import annotations

from pathlib import Path

from app.services.language.package.checks import as_list, as_object, read_json, require


def state(path: Path) -> None:
    """Check one state file: its JSON shape, its record types and its required text."""
    kind = path.name.split("-", 1)[0]
    value = read_json(path)
    if kind in ("runs", "feedback"):
        rows(
            value,
            {
                "question": (str,),
                "answer": (str,),
                "correction": (str,),
                "correct": (bool,),
                "learned": (bool, type(None)),
            },
            path.name,
        )
        return
    row = as_object(value, path.name)
    if kind == "tenant":
        _tenant(row)
    elif kind == "expressions":
        _expressions(row)
    elif kind == "progress":
        _progress(row)
    elif kind in ("model", "level"):
        fields(row, {"name": (str,), "start_level": (str,), "approved_level": (str,)}, path.name)
        rows(row.get("history", []), {"level": (str,), "approved_at": (str,)}, "Level history")


def _expressions(row: dict) -> None:
    for item in as_list(row.get("items"), dict, "Expressions items"):
        require(
            all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("id", "phrase", "reply")
            ),
            "Expression needs id, phrase and reply",
        )
        require(
            item.get("category") in ("open", "close", "pause", "resume"),
            "Unknown expression category",
        )


def _progress(row: dict) -> None:
    fields(row, {"steps_done": (int,), "steps_total": (int,), "started": (int, float)}, "Progress")
    stages = as_object(row.get("stages", {}), "Progress stages")
    require(
        all(value in ("waiting", "working", "done") for value in stages.values()),
        "Invalid progress stage status",
    )


def _tenant(row: dict) -> None:
    fields(
        row, {"letters": (str,), "active_test": (dict, type(None)), "tenant_id": (int,)}, "Learner"
    )
    for key in ("passed_stages", "accepted_duplicates"):
        as_list(row.get(key, []), str, f"Learner {key}")
    cards = {
        "text": (str,),
        "word": (str,),
        "meaning": (str,),
        "question": (str,),
        "answer": (str,),
        "subjects": (list,),
    }
    for key in ("lessons", "word_cards", "lesson_cards", "test_history"):
        rows(row.get(key, []), cards, f"Learner {key}")
    for key, required in (
        ("lessons", ("text",)),
        ("word_cards", ("word", "meaning")),
        ("lesson_cards", ("question", "answer")),
    ):
        for item in row.get(key, []):
            require(
                all(isinstance(item.get(field), str) for field in required),
                f"Learner {key} is missing required text",
            )


def fields(row: dict, types: dict, label: str) -> None:
    """Check the type of every field that is present, and only those."""
    for key, expected in types.items():
        if key in row:
            require(type(row[key]) in expected, f"{label}.{key} has the wrong type")


def rows(value: object, types: dict, label: str) -> None:
    """Check the types of every row in a list of records."""
    for row in as_list(value, dict, label):
        fields(row, types, label)
