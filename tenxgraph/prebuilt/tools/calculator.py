"""Safe arithmetic tools for AgentFlow agents."""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any

from agentflow.utils.decorators import tool


_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 500
_MAX_ABS_VALUE = 10**12
_MAX_POWER_EXPONENT = 12


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        value = node.value
        if not math.isfinite(float(value)) or abs(value) > _MAX_ABS_VALUE:
            raise ValueError("numeric value is outside the allowed range")
        return value

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return op(_evaluate_node(node.operand))

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported binary operator")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError("power exponent is outside the allowed range")
        result = op(left, right)
        if not math.isfinite(float(result)) or abs(result) > _MAX_ABS_VALUE:
            raise ValueError("result is outside the allowed range")
        return result

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool(
    name="safe_calculator",
    description=(
        "Safely evaluate a basic arithmetic expression. Supports numbers, parentheses, "
        "and +, -, *, /, //, %, and ** with conservative size limits."
    ),
    tags=["math", "calculator"],
    capabilities=["calculate"],
)
def safe_calculator(expression: str, precision: int | None = None) -> str:
    """Evaluate a basic arithmetic expression safely."""
    if not expression or not expression.strip():
        return json.dumps({"error": "expression is required"})
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return json.dumps({"error": "expression is too long"})

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree)
        if precision is not None and isinstance(result, float):
            safe_precision = max(0, min(int(precision), 12))
            result = round(result, safe_precision)
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
