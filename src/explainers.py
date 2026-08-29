"""Generate human-friendly explanations for model predictions."""


def explain_prediction(pred_label: str, model_prob: float = None) -> str:
    explanation_map = {
        'SyntaxError': 'There is a syntax problem: code cannot be parsed. Check punctuation and structure.',
        'NameError': 'A variable or function name is used before it is defined.',
        'TypeError': 'An operation received an operand of an unexpected type.',
        'IndexError': 'A sequence was accessed with an invalid index.',
        'IndentationError': 'Indentation is inconsistent or incorrect (Python).',
        'ZeroDivisionError': 'Division by zero was attempted.',
        'AttributeError': 'An attribute or method was accessed on an object that does not have it.',
        'Other': 'The error is uncommon or could not be classified confidently.'
    }
    base = explanation_map.get(pred_label, explanation_map['Other'])
    if model_prob is not None:
        base += f" (model confidence: {model_prob:.2f})"
    return base
