"""Produce suggested code fixes and debugging recommendations.

Recommendations are derived only from static warnings detected by the analyzer.
They are conservative, actionable hints rather than guaranteed fixes.
"""
from typing import List, Dict, Any


def recommendations_from_warnings(warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn analyzer warnings into concrete recommendations.

    Each recommendation is a dict with fields: `type`, `message`, `suggestion`,
    and optional `line` and `severity` copied from the warning.
    """
    recs: List[Dict[str, Any]] = []
    for w in warnings:
        t = w.get('type')
        base = {
            'type': t,
            'severity': w.get('severity'),
            'line': w.get('line'),
            'message': w.get('message'),
            'suggestion': w.get('suggestion') or 'Review the code and fix the reported issue.'
        }
        # Minor customization for a few known types
        if t == 'syntax_error' or t == 'unmatched_delimiter':
            base['suggestion'] = 'Check the reported line for missing punctuation, brackets, or invalid Python syntax.'
        elif t == 'indentation_tabs' or t == 'indentation_mixed':
            base['suggestion'] = 'Use consistent spaces for indentation (PEP8 recommends 4 spaces). Consider running a formatter.'
        elif t == 'bare_except':
            base['suggestion'] = 'Replace bare except with specific exception types (e.g., "except ValueError:").'
        elif t == 'dynamic_execution':
            base['suggestion'] = 'Avoid exec/eval; evaluate safer design patterns and validate input before executing dynamic code.'
        recs.append(base)
    return recs


def unique_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return recommendations with repeated displayed suggestions removed.

    Different static findings can legitimately result in the same next step.
    The UI should show that text once while preserving the order of the first
    matching recommendation.
    """
    unique: List[Dict[str, Any]] = []
    seen_suggestions = set()
    for recommendation in recommendations:
        suggestion = recommendation.get('suggestion', '')
        if suggestion not in seen_suggestions:
            unique.append(recommendation)
            seen_suggestions.add(suggestion)
    return unique


def suggest_fix(code: str, error_type: str) -> str:
    """Backward-compatible simple suggester for tests or old callers."""
    mapping = {
        'SyntaxError': 'Check for missing colons, parentheses, or incorrect indentation.',
        'NameError': 'Ensure the variable or function is defined before use and check for typos.',
        'TypeError': 'Check data types and conversions; add explicit casts if needed.',
        'IndexError': 'Validate indices and check sequence lengths before accessing.',
        'IndentationError': 'Fix inconsistent indentation (spaces vs tabs) and block alignment.',
        'ZeroDivisionError': 'Guard divisions with checks for zero denominators.',
        'AttributeError': 'Verify the object type and available attributes/methods.'
    }
    return mapping.get(error_type, 'Review the code and add logging or prints to isolate the issue.')
