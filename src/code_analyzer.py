"""Static analysis helpers that operate without executing submitted code.

This module provides a simple, explainable static analyzer that builds on
`ast` and token-level inspection. It purposely avoids any form of execution
(`exec`, `eval`, subprocess, importlib execution, etc.). The output is a
serializable dictionary suitable for UI/display layers and automated tests.
"""
from typing import Any, Dict, List, Optional
import ast
import io
import tokenize

from src.feature_extraction import extract_features


def _detect_unmatched_delimiters(code: str) -> List[Dict[str, Any]]:
    """Detect delimiter mismatches from Python tokens, ignoring strings/comments."""
    warnings: List[Dict[str, Any]] = []
    opening = {'(': ')', '[': ']', '{': '}'}
    closing = {value: key for key, value in opening.items()}
    stack = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in opening:
                stack.append((token.string, token.start[0]))
            elif token.string in closing:
                if stack and stack[-1][0] == closing[token.string]:
                    stack.pop()
                else:
                    warnings.append({'type': 'unmatched_delimiter', 'severity': 'warning',
                                     'message': f'Unmatched closing delimiter "{token.string}".', 'line': token.start[0],
                                     'explanation': 'This closing delimiter has no matching opening delimiter.',
                                     'suggestion': 'Check that every opening delimiter has a matching closing delimiter.'})
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    for delimiter, line in stack:
        warnings.append({'type': 'unmatched_delimiter', 'severity': 'warning',
                         'message': f'Mismatched delimiter: "{delimiter}" is not closed.', 'line': line,
                         'explanation': 'A Python delimiter was opened but no matching closing delimiter was found.',
                         'suggestion': 'Check that every opening delimiter has a matching closing delimiter.'})
    return warnings

def _detect_indentation_issues(code: str) -> List[Dict[str, Any]]:
    """Detect obvious indentation issues such as tabs or a mixture of tabs and spaces.

    This is conservative: it reports presence of tab characters and whether both
    tabs and leading-space indents are present across non-blank lines.
    """
    warnings: List[Dict[str, Any]] = []
    lines = code.splitlines()
    has_tab = False
    for ln in lines:
        if not ln.strip():
            continue
        if '\t' in ln:
            has_tab = True

    if has_tab:
        warnings.append({
            'type': 'indentation_tabs',
            'severity': 'warning',
            'message': 'Tab character(s) used for indentation.',
            'line': None,
            'explanation': 'Tabs are present in the source; mixing tabs and spaces may cause IndentationError.',
            'suggestion': 'Use spaces consistently (PEP8 recommends 4 spaces) or convert tabs to spaces.'
        })

    # detect mixed leading whitespace (lines that start with spaces and lines that start with tabs)
    starts_with_space = any(ln.startswith(' ') for ln in lines if ln.strip())
    starts_with_tab = any(ln.startswith('\t') for ln in lines if ln.strip())
    if starts_with_space and starts_with_tab:
        warnings.append({
            'type': 'indentation_mixed',
            'severity': 'warning',
            'message': 'Mixed indentation detected (tabs and spaces).',
            'line': None,
            'explanation': 'Some lines are indented with spaces while others use tabs.',
            'suggestion': 'Normalize indentation to spaces or tabs consistently.'
        })

    return warnings


def _detect_suspicious_structures(code: str) -> List[Dict[str, Any]]:
    """Detect a small set of structural patterns reliably via the AST.

    Examples: bare except, use of eval/exec (flagged as potentially risky),
    and very large functions (informational).
    """
    warnings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return warnings

    for node in ast.walk(tree):
        # bare except
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                warnings.append({
                    'type': 'bare_except',
                    'severity': 'warning',
                    'message': 'Bare except clause detected.',
                    'line': getattr(node, 'lineno', None),
                    'explanation': 'A bare "except:" catches all exceptions and can hide errors.',
                    'suggestion': 'Catch specific exception types (e.g., "except ValueError:").'
                })
        # exec/eval usage
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ('exec', 'eval'):
                warnings.append({
                    'type': 'dynamic_execution',
                    'severity': 'warning',
                    'message': f'Use of {node.func.id}() detected.',
                    'line': getattr(node, 'lineno', None),
                    'explanation': f'Calling {node.func.id}() executes dynamic code which is risky.',
                    'suggestion': 'Avoid exec/eval; refactor to safer alternatives.'
                })
        # very large function body (informational)
        if isinstance(node, ast.FunctionDef):
            body_len = len(node.body)
            if body_len > 200:
                warnings.append({
                    'type': 'large_function',
                    'severity': 'info',
                    'message': f'Function "{node.name}" has a large body ({body_len} statements).',
                    'line': getattr(node, 'lineno', None),
                    'explanation': 'Very large functions are harder to reason about and test.',
                    'suggestion': 'Consider refactoring into smaller helper functions.'
                })

    return warnings


def analyze_source(code: str) -> Dict[str, Any]:
    """Perform static analysis on `code` and return a structured result.

    The result contains:
      - parse_success: bool
      - syntax_error: optional dict with type/message/line
      - warnings: list of structured warnings
      - metrics: feature dict returned by `extract_features`

    This function never executes the submitted code.
    """
    result: Dict[str, Any] = {}
    if code is None:
        code = ''

    # Basic metrics via feature_extraction (reuses training-time logic)
    try:
        metrics = extract_features(code)
    except Exception:
        metrics = {}
    result['metrics'] = metrics

    # Attempt to parse and capture a SyntaxError with line info if present
    try:
        ast.parse(code)
        result['parse_success'] = True
        result['syntax_error'] = None
    except SyntaxError as se:
        result['parse_success'] = False
        result['syntax_error'] = {
            'type': type(se).__name__,
            'message': str(se.msg) if hasattr(se, 'msg') else str(se),
            'line': se.lineno,
            'offset': se.offset,
        }
    except Exception as e:
        # Other parse-time errors
        result['parse_success'] = False
        result['syntax_error'] = {
            'type': type(e).__name__,
            'message': str(e),
            'line': None,
        }

    # Collect warnings from multiple detectors (conservative)
    warnings: List[Dict[str, Any]] = []
    warnings.extend(_detect_unmatched_delimiters(code))
    warnings.extend(_detect_indentation_issues(code))
    warnings.extend(_detect_suspicious_structures(code))

    # If parse failed and no specific syntax warning present, add a generic one
    if not result.get('parse_success', True) and not any(w.get('type') == 'syntax_error' for w in warnings):
        syntax = result.get('syntax_error')
        warnings.insert(0, {
            'type': 'syntax_error',
            'severity': 'error',
            'message': 'Python syntax could not be parsed.',
            'line': syntax.get('line') if syntax else None,
            'explanation': 'ast.parse() failed; code contains a syntax error.',
            'suggestion': 'Check the reported line and surrounding context for missing punctuation or unmatched delimiters.'
        })

    result['warnings'] = warnings
    return result
