"""Age, date, and number answers that do not need the learned lesson file."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

from app.services.language.curriculum import ASSISTANT_BIRTHDAY
from app.services.language.patterns import _NEXT_NUMBER_REQUEST

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS_NAMES = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def answer_assistant_age(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.casefold())
    normalized = " ".join(normalized.split())
    if not re.search(r"\b(?:your age|how old are you|what age are you|old are you)\b", normalized):
        return None
    today = date.today()
    years = (
        today.year
        - ASSISTANT_BIRTHDAY.year
        - ((today.month, today.day) < (ASSISTANT_BIRTHDAY.month, ASSISTANT_BIRTHDAY.day))
    )
    if years == 0:
        return (
            "My birthday is 19 October 2026, so I’m 0 years old today—I was born today!"
            if today == ASSISTANT_BIRTHDAY
            else "My birthday is 19 October 2026, so I’m not born yet."
        )
    return f"My birthday is 19 October 2026, so I’m {years} year{'s' if years != 1 else ''} old."


def answer_current_date_time(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.casefold().replace("’", "").replace("'", ""))
    normalized = " ".join(normalized.split())
    now = datetime.now(UTC).astimezone()
    if re.search(
        r"\b(?:what time is it|what is the time|whats the time|tell me the time|current time"
        r"|time right now|time now|what time is it now)\b",
        normalized,
    ):
        time_text = now.strftime("%I:%M %p").lstrip("0")
        timezone_name = now.tzname()
        suffix = f" {timezone_name}" if timezone_name else ""
        return f"It is {time_text}{suffix} on the app’s local clock."
    if re.search(r"\b(?:what month is it|what is the current month|current month)\b", normalized):
        return f"It is {now.strftime('%B')}."
    if re.search(r"\b(?:what year is it|what is the current year|current year)\b", normalized):
        return f"It is {now.year}."
    if re.search(
        r"\b(?:what is today|what day is it|what day is today|what is the date|what date is it"
        r"|what date is today|whats todays date|what is todays date|todays date|current date"
        r"|todays day)\b",
        normalized,
    ):
        date_text = now.strftime("%A, %d %B %Y").replace(" 0", " ")
        return f"Today is {date_text}."
    return None


def _number_value(raw: str) -> int | None:
    token = raw.casefold().strip()
    if token.isdecimal():
        return int(token)
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    if "-" in token:
        parts = token.split("-", 1)
        if (
            len(parts) == 2
            and parts[0] in _TENS_NAMES
            and parts[1] in _NUMBER_WORDS
            and 1 <= _NUMBER_WORDS[parts[1]] <= 9
        ):
            return _NUMBER_WORDS[parts[1]] + int(
                next(key for key, name in _TENS_NAMES.items() if name == parts[0])
            )
    return None


def _number_name(value: int) -> str:
    if value == 100:
        return "one hundred"
    if value < 20:
        return _NUMBER_NAMES[value]
    tens, ones = divmod(value, 10)
    return _TENS_NAMES[tens * 10] + (f"-{_NUMBER_NAMES[ones]}" if ones else "")


def _answer_number_sequence_question(question: str) -> str | None:
    """Answer "what comes after 20?" for any number, not only the taught ones."""
    match = _NEXT_NUMBER_REQUEST.search(question.strip())
    if not match:
        return None
    after_token, before_token = match.group(1), match.group(2)
    value = _number_value(after_token or before_token or "")
    if value is None or not (0 <= value <= 100):
        return None
    if after_token:
        return f"The number after {value} is {value + 1}."
    return f"The number before {value} is {value - 1}."


def _answer_counting_question(question: str) -> str | None:
    q = question.strip().casefold().rstrip("?.! ")
    match = re.fullmatch(r"count\s+(?:numbers\s+)?(?:from\s+)?([\w-]+)\s*(?:to|[-–])\s*([\w-]+)", q)
    if match:
        start_text, end_text = match.group(1), match.group(2)
        start, end = (_number_value(start_text), _number_value(end_text))
    else:
        match = re.fullmatch(r"count\s+(?:numbers\s+)?(?:up\s+)?to\s+([\w-]+)", q)
        if not match:
            return None
        start, end_text = 1, match.group(1)
        start_text = "one"
        end = _number_value(end_text)
    if (
        start is None
        or end is None
        or not (0 <= start <= 100 and 0 <= end <= 100)
        or abs(end - start) > 100
    ):
        return None
    step = 1 if end >= start else -1
    use_words = start_text.isalpha() and end_text.isalpha()
    render = _number_name if use_words else str
    sequence = ", ".join(render(value) for value in range(start, end + step, step))
    return f"Counting from {render(start)} to {render(end)}: {sequence}."


def _answer_number_spelling_question(question: str) -> str | None:
    q = question.strip().casefold().rstrip("?.! ")
    patterns = (
        r"spell\s+(?:the\s+)?(?:number\s+)?([\w-]+)",
        r"(?:write|say)\s+([\w-]+)\s+in\s+words",
        r"what\s+is\s+([\w-]+)\s+in\s+words",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            value = _number_value(match.group(1))
            if value is not None and 0 <= value <= 100:
                return f"{_number_name(value)}."
    return None
