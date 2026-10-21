"""Safe arithmetic answers with no eval or calculator service."""

from __future__ import annotations

import math
import re


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _calculation_result(operation: str, left: float, right: float) -> str | None:
    try:
        if operation == "plus":
            return f"{_format_number(left)} plus {_format_number(right)} is {_format_number(left + right)}."
        if operation == "minus":
            return f"{_format_number(left)} minus {_format_number(right)} is {_format_number(left - right)}."
        if operation == "times":
            return f"{_format_number(left)} times {_format_number(right)} is {_format_number(left * right)}."
        if operation == "divided":
            if right == 0:
                return "You cannot divide a number by zero."
            return (
                f"{_format_number(left)} divided by {_format_number(right)} "
                f"is {_format_number(left / right)}."
            )
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _calculation_special(text: str) -> str | None:
    root = re.search(r"(?:square\s*root|sqrt)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if root:
        value = float(root.group(1))
        if value < 0:
            return "A negative number does not have a real square root."
        answer = math.sqrt(value)
        return f"The square root of {_format_number(value)} is {_format_number(answer)}."
    squared = re.search(r"(-?\d+(?:\.\d+)?)\s+(?:squared|to the power of 2)\b", text)
    if squared:
        value = float(squared.group(1))
        return f"{_format_number(value)} squared is {_format_number(value * value)}."
    cubed = re.search(r"(-?\d+(?:\.\d+)?)\s+cubed\b", text)
    if cubed:
        value = float(cubed.group(1))
        return f"{_format_number(value)} cubed is {_format_number(value**3)}."
    percent = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s+(-?\d+(?:\.\d+)?)", text)
    if percent:
        value = float(percent.group(1))
        total = float(percent.group(2))
        return (
            f"{_format_number(value)} percent of {_format_number(total)} "
            f"is {_format_number(total * value / 100)}."
        )
    return None


def _calculation_symbol(text: str) -> str | None:
    symbol = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)", text)
    if not symbol:
        return None
    operation = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "x": "times",
        "×": "times",
        "/": "divided",
        "÷": "divided",
    }.get(symbol.group(2))
    if not operation:
        return None
    return _calculation_result(operation, float(symbol.group(1)), float(symbol.group(3)))


def _calculation_words(text: str) -> str | None:
    words = re.search(
        r"(-?\d+(?:\.\d+)?)\s+(plus|add|added to|addition of|minus|subtract|subtracted from"
        r"|take away|times|multiplied by|multiply|divided by|divide|divided into)\s+"
        r"(-?\d+(?:\.\d+)?)",
        text,
    )
    if not words:
        return None
    left, right, operator = float(words.group(1)), float(words.group(3)), words.group(2)
    if operator in {"plus", "add", "added to", "addition of"}:
        return _calculation_result("plus", left, right)
    if operator in {"minus", "subtract", "take away"}:
        return _calculation_result("minus", right, left) if operator != "minus" else _calculation_result("minus", left, right)
    if operator == "subtracted from":
        return _calculation_result("minus", right, left)
    if operator in {"times", "multiplied by", "multiply"}:
        return _calculation_result("times", left, right)
    if operator in {"divided by", "divide"}:
        return _calculation_result("divided", left, right)
    if operator == "divided into":
        return _calculation_result("divided", right, left)
    return None


def _calculation_phrases(text: str) -> str | None:
    patterns = (
        (r"divide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", "divided", False),
        (r"(?:add|plus)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)", "plus", False),
        (r"(?:subtract|take)\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)", "minus", True),
        (r"multiply\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", "times", False),
    )
    for pattern, operation, swapped in patterns:
        match = re.search(pattern, text)
        if match:
            left, right = float(match.group(1)), float(match.group(2))
            return _calculation_result(operation, right, left) if swapped else _calculation_result(operation, left, right)
    doubled = re.search(r"double\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", text)
    if doubled:
        value = float(doubled.group(1))
        return f"Double of {_format_number(value)} is {_format_number(value * 2)}."
    tripled = re.search(r"triple\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", text)
    if tripled:
        value = float(tripled.group(1))
        return f"Triple of {_format_number(value)} is {_format_number(value * 3)}."
    halved = re.search(r"half\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if halved:
        value = float(halved.group(1))
        return f"Half of {_format_number(value)} is {_format_number(value / 2)}."
    return None


def _answer_calculation_question(question: str) -> str | None:
    """Answer adding, subtraction, multiplication, division, and square roots."""
    text = question.strip().lower().rstrip("?.! ")
    if not text:
        return None
    checks = (
        _calculation_special(text),
        _calculation_symbol(text),
        _calculation_words(text),
        _calculation_phrases(text),
    )
    return next((answer for answer in checks if answer), None)

