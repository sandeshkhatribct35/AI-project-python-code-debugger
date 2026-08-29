"""Static analysis and safe execution helpers."""
import ast
from typing import Tuple, Optional


def analyze_code_safely(code: str) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to parse the code and return (exception_type, exception_message).

    This function currently uses `ast.parse` for safe syntax checks. Do NOT execute untrusted code.
    """
    try:
        ast.parse(code)
    except Exception as e:
        return (type(e).__name__, str(e))
    return (None, None)
