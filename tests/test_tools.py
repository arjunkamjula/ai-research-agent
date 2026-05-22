"""tests/test_tools.py — unit tests for individual tools."""

from app.tools.calculator import calculator
from app.tools.python_executor import python_executor


def test_calculator_basic():
    assert calculator("2 + 2") == "4"


def test_calculator_float():
    result = calculator("10 / 3")
    assert result.startswith("3.333")


def test_calculator_expression():
    assert calculator("(42 * 2) + 10") == "94"


def test_calculator_division_by_zero():
    result = calculator("1 / 0")
    assert "zero" in result.lower()


def test_calculator_blocked_function():
    result = calculator("__import__('os')")
    assert "Error" in result or "Unsupported" in result


def test_python_executor_basic():
    result = python_executor("print(1 + 1)")
    assert "2" in result


def test_python_executor_blocked():
    result = python_executor("import os; print(os.listdir('.'))")
    assert "blocked" in result.lower()


def test_python_executor_timeout():
    result = python_executor("while True: pass")
    assert "timed out" in result.lower()


def test_python_executor_math():
    result = python_executor("print(sum(range(100)))")
    assert "4950" in result
