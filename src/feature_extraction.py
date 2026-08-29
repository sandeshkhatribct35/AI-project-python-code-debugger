"""Feature extraction from source code and error messages."""
from typing import Any, Dict


def extract_features(code: str, error_message: str = None) -> Dict[str, Any]:
    """Return a feature dict for a single sample.

    Start with simple features: length, number of lines, token counts, presence of keywords.
    Replace with AST-based features later.
    """
    features = {}
    features['n_chars'] = len(code)
    features['n_lines'] = code.count('\n') + 1
    features['has_division'] = '/' in code
    features['has_indexing'] = '[' in code and ']' in code
    if error_message:
        features['error_msg_len'] = len(error_message)
    else:
        features['error_msg_len'] = 0
    return features
