"""Produce suggested code fixes and debugging recommendations."""


def suggest_fix(code: str, error_type: str) -> str:
    """Return a simple suggested fix or guidance based on error_type. These are templates, not guaranteed fixes."""
    suggestions = {
        'SyntaxError': 'Check for missing colons, parentheses, or incorrect indentation.',
        'NameError': 'Ensure the variable or function is defined before use and check for typos.',
        'TypeError': 'Check data types and conversions; add explicit casts if needed.',
        'IndexError': 'Validate indices and check sequence lengths before accessing.',
        'IndentationError': 'Fix inconsistent indentation (spaces vs tabs) and block alignment.',
        'ZeroDivisionError': 'Guard divisions with checks for zero denominators.',
        'AttributeError': 'Verify the object type and available attributes/methods.'
    }
    return suggestions.get(error_type, 'Review the code and add logging or prints to isolate the issue.')
