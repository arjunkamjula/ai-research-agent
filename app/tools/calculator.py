"""
app/tools/calculator.py

Safe arithmetic evaluator using Python's ast module.
No exec() or eval() — parses the expression tree and
evaluates only allowed node types.
"""

import ast
import operator

ALLOWED_OPERATORS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}

ALLOWED_FUNCTIONS = {
    "abs":   abs,
    "round": round,
    "min":   min,
    "max":   max,
    "sum":   sum,
    "int":   int,
    "float": float,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fn = ALLOWED_FUNCTIONS.get(node.func.id)
        if not fn:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_eval_node(a) for a in node.args]
        return fn(*args)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Math expression e.g. '(42 * 1.15) / 3'

    Returns:
        Result as string, or error message
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"

    try:
        tree   = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        if isinstance(result, float):
            return str(round(result, 10))
        return str(result)

    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
