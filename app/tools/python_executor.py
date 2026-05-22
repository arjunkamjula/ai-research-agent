"""
app/tools/python_executor.py

Sandboxed Python code execution via subprocess.
Captures stdout and stderr. Hard timeout prevents runaway processes.
Dangerous builtins are blocked at the code level.
"""

import subprocess
import sys
import textwrap

BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
]

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 3000


def python_executor(code: str) -> str:
    """
    Execute Python code in a subprocess and return output.

    Blocks dangerous operations (file system access, subprocess spawning).
    Hard timeout of 10 seconds.

    Args:
        code: Python code to execute — must use print() for output

    Returns:
        Combined stdout + stderr from execution
    """
    code_lower = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return (
                f"Execution blocked: '{pattern}' is not allowed. "
                "Only pure computation is permitted."
            )

    wrapped = textwrap.dedent(f"""
import math
import json
import re
import datetime
from collections import defaultdict, Counter

{code}
""")

    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output = True,
            text           = True,
            timeout        = TIMEOUT_SECONDS,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        if not output.strip():
            output = "(no output — use print() to display results)"

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} chars)"

        return output.strip()

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as e:
        return f"Execution error: {e}"
