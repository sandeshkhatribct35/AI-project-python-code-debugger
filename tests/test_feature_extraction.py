from src.feature_extraction import extract_features


def test_extract_basic():
    code = "print(1)"
    feats = extract_features(code)
    assert 'n_chars' in feats
    assert 'n_lines' in feats


def test_valid_python_function():
    code = """
def foo(x):
    # simple function
    return x + 1
"""
    f = extract_features(code)
    assert f['ast_parse_success'] == 1
    assert f['n_functions'] >= 1


def test_invalid_python_code():
    code = "def broken(\n  return 1"
    f = extract_features(code)
    assert f['ast_parse_success'] == 0
    assert f['ast_syntax_error'] == 1


def test_empty_code():
    f = extract_features("")
    assert f['n_chars'] == 0
    assert f['n_lines'] == 0


def test_multiple_functions_and_comments():
    code = """
def a():
    pass

def b():
    # comment here
    return 2
"""
    f = extract_features(code)
    assert f['n_functions'] >= 2
    assert f['n_comments'] >= 1


def test_indentation_and_tabs():
    code = "def t():\n\tprint(1)\n    print(2)"
    f = extract_features(code)
    assert f['has_tab_indentation'] in (0,1)
    assert 'avg_leading_spaces' in f
