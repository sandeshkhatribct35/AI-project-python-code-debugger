from src.feature_extraction import extract_features


def test_extract_basic():
    code = "print(1)"
    feats = extract_features(code)
    assert 'n_chars' in feats
    assert 'n_lines' in feats
