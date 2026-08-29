import pytest

from src import code_analyzer


def test_valid_python_parses():
    code = """
def add(a, b):
    return a + b
"""
    out = code_analyzer.analyze_source(code)
    assert out['parse_success'] is True
    assert out['syntax_error'] is None
    assert isinstance(out['metrics'], dict)


def test_invalid_python_detected():
    code = """
def add(a, b)
    return a + b
"""
    out = code_analyzer.analyze_source(code)
    assert out['parse_success'] is False
    # should include a syntax_error in warnings
    types = [w['type'] for w in out['warnings']]
    assert 'syntax_error' in types or out['syntax_error'] is not None


def test_unmatched_delimiter():
    code = "a = (1 + 2"
    out = code_analyzer.analyze_source(code)
    types = [w['type'] for w in out['warnings']]
    assert 'unmatched_delimiter' in types


def test_tab_indentation_detection():
    code = "def f():\n\treturn 1\n"
    out = code_analyzer.analyze_source(code)
    types = [w['type'] for w in out['warnings']]
    assert 'indentation_tabs' in types or 'indentation_mixed' in types
