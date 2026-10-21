"""Read labelled fields from PDF pages and answer questions about them."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_FIELD_KEYS = {
    "customer name": "customer_name",
    "agent": "agent",
    "mobile no": "mobile_no",
    "email": "email",
    "reservation code": "reservation_code",
    "status": "status",
    "voucher no": "voucher_no",
    "booking date": "booking_date",
    "policy code": "policy_code",
    "journey date": "journey_date",
}
_FIELD_LABELS = {
    "customer_name": "customer name",
    "agent": "agent",
    "mobile_no": "mobile no",
    "email": "email",
    "reservation_code": "reservation code",
    "status": "status",
    "voucher_no": "voucher no",
    "booking_date": "booking date",
    "policy_code": "policy code",
    "journey_date": "journey date",
}
_QUERY_FIELDS = (
    ("booking_date", ("booking date",)),
    ("journey_date", ("journey date",)),
    ("policy_code", ("policy code",)),
    ("reservation_code", ("reservation code", "reservation")),
    ("customer_name", ("customer name", "name of the customer", "customer named")),
    ("mobile_no", ("mobile number", "mobile no", "mobile")),
    ("voucher_no", ("voucher number", "voucher no")),
    ("agent", ("agent",)),
    ("email", ("email",)),
    ("status", ("status",)),
)
_DATE_KEYS = ("booking_date", "journey_date")


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _field_key(label: str) -> str | None:
    normalized = " ".join(_normalized_tokens(label))
    return _FIELD_KEYS.get(normalized)


def _line_text(words: list[tuple[Any, ...]]) -> str:
    return " ".join(str(word[4]) for word in sorted(words, key=lambda item: item[0]))


def _group_lines(words: list[tuple[Any, ...]]) -> list[list[tuple[Any, ...]]]:
    lines: list[list[tuple[Any, ...]]] = []
    for word in sorted(words, key=lambda item: (round(float(item[1]) / 4), item[0])):
        if lines and abs(float(lines[-1][0][1]) - float(word[1])) <= 4:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _join_wrapped(lines: list[str]) -> str:
    result = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if result.endswith("-"):
            result = result[:-1] + line
        elif result:
            result = f"{result} {line}"
        else:
            result = line
    return result.strip()


def _column_value(
    lines: list[list[tuple[Any, ...]]],
    start_x: float,
    end_x: float,
    heading_y: float,
) -> str:
    values: list[str] = []
    for line in lines:
        word_y = float(line[0][1])
        if not heading_y < word_y <= heading_y + 45:
            continue
        selected = [
            word for word in line if start_x - 4 <= float(word[0]) < max(end_x - 2, start_x + 1)
        ]
        if selected:
            values.append(_line_text(selected))
    return _join_wrapped(values)


def _heading_spans(
    line: list[tuple[Any, ...]],
) -> list[tuple[str, float, float]]:
    words = sorted((word for word in line if str(word[4]).strip()), key=lambda item: item[0])
    spans: list[tuple[str, float, float]] = []
    index = 0
    while index < len(words):
        matched = False
        for size in (3, 2, 1):
            chunk = words[index : index + size]
            if len(chunk) != size:
                continue
            key = _field_key(" ".join(str(word[4]) for word in chunk))
            if key is None or not all(str(word[4]).isupper() for word in chunk):
                continue
            spans.append((key, float(chunk[0][0]), float(chunk[0][1])))
            index += size
            matched = True
            break
        if not matched:
            index += 1
    return spans


def _extract_pdf_field_values(page: Any, page_text: str = "") -> dict[str, str]:
    """Return labelled field values found on a PDF page."""
    fields: dict[str, str] = {}
    words = page.get_text("words")
    lines = _group_lines(words)
    for line in lines:
        heading_spans = _heading_spans(line)
        for index, (key, start_x, heading_y) in enumerate(heading_spans):
            end_x = (
                heading_spans[index + 1][1] if index + 1 < len(heading_spans) else 10_000.0
            )
            value = _column_value(lines, start_x, end_x, heading_y)
            if value:
                fields[key] = value
    for line in page_text.splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z .]*?)\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        key = _field_key(match.group(1))
        if key is not None:
            fields[key] = match.group(2).strip()
    return fields


def _requested_fields(question: str) -> list[str]:
    lowered = question.casefold()
    if re.search(r"\bdate\b", lowered) and not any(
        term in lowered for key, terms in _QUERY_FIELDS if key in _DATE_KEYS for term in terms
    ):
        return list(_DATE_KEYS)
    matches: list[tuple[int, str]] = []
    for key, terms in _QUERY_FIELDS:
        for term in terms:
            if term in lowered:
                matches.append((len(term), key))
    ordered: list[str] = []
    for _, key in sorted(matches, reverse=True):
        if key not in ordered:
            ordered.append(key)
    return ordered


def _display_label(key: str) -> str:
    if key == "voucher_no":
        return "Voucher no"
    return key.replace("_", " ").capitalize()


def _unclear(key: str) -> str:
    label = _FIELD_LABELS.get(key, key.replace("_", " "))
    return (
        f"I couldn't reliably identify the {label} in this document. "
        "Please check the source passage."
    )


def _field_values(hits: list[dict], key: str) -> set[str]:
    values: set[str] = set()
    for hit in hits:
        metadata = hit.get("metadata") or {}
        import json

        try:
            fields = json.loads(metadata.get("pdf_fields") or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(fields, dict):
            value = str(fields.get(key, "")).strip()
            if value:
                values.add(value)
    return values


def _answer_field(hits: list[dict], key: str) -> str | None:
    values = _field_values(hits, key)
    if len(values) != 1:
        return None if not values else _unclear(key)
    return f"{_display_label(key)}: {next(iter(values))}"


def _snippet(text: str, question: str) -> str:
    question_terms = {
        token
        for token in _normalized_tokens(question)
        if token not in {"what", "which", "where", "when", "who", "the", "a", "an", "is", "are"}
    }
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    best = max(
        sentences,
        key=lambda sentence: len(question_terms & set(_normalized_tokens(sentence))),
        default="",
    )
    if not best:
        return text.strip()[:240]
    return best[:260].strip()


def answer_from_document(question: str, hits: list[dict]) -> str:
    """Answer from PDF field metadata, falling back to a short evidence snippet."""
    requested = _requested_fields(question)
    if len(requested) > 1 and requested == list(_DATE_KEYS):
        available = [key for key in requested if _field_values(hits, key)]
        if len(available) > 1:
            return (
                "I found multiple possible fields for the date in this document. "
                "Please check the source passage."
            )
    for key in requested:
        if not _field_values(hits, key):
            continue
        answer = _answer_field(hits, key)
        if answer is not None:
            return answer
        return _unclear(key)
    if requested:
        return _unclear(requested[0])
    text = str(hits[0].get("text", "")) if hits else ""
    return _snippet(text, question)


def faq_field_subjects_compatible(first: str, second: str) -> bool:
    """Whether two FAQ questions ask about the same document field."""
    first_fields = set(_requested_fields(first))
    second_fields = set(_requested_fields(second))
    if not first_fields or not second_fields:
        return True
    return bool(first_fields & second_fields)
