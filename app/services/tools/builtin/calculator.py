"""Calculator tool: evaluate a math expression safely.

The calculator uses ast.literal_eval to parse the expression, which only
allows literals and basic operators. No eval, no arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from app.services.tools.registry import Tool, register

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a math expression and return the result.

    Only numeric literals and the operators +, -, *, /, //, %, ** are allowed.
    Anything else raises ValueError.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid expression: {error}") from error
    result = _eval_node(tree.body)
    return {"result": result}


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _OPERATORS[op_type](operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


register(
    Tool(
        name="calculator",
        description="Evaluate a math expression. Supports +, -, *, /, //, %, **.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
        requires_confirmation=False,
        handler=calculate,
    )
)
