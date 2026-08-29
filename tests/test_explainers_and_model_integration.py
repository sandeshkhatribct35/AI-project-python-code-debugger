from src import explainers


def test_analyze_and_explain_runs_for_valid_code():
    code = "def add(a, b):\n    return a + b\n"
    out = explainers.analyze_and_explain(code)
    assert 'static' in out
    assert 'prediction' in out
    assert 'explanation' in out


def test_analyze_and_explain_handles_syntax_error():
    code = "def add(a, b)\n    return a + b\n"
    out = explainers.analyze_and_explain(code)
    assert out['static']['parse_success'] is False
    # Explanation should mention no static issues or syntax clearly
    assert isinstance(out['explanation'], str)
