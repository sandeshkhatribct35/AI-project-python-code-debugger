"""Explanation layer combining model results and static analysis.

This module provides a safe backend function `analyze_and_explain` which:
 - applies conservative unescaping,
 - extracts the trained model features in the saved order,
 - calls the saved model for prediction/probability,
 - runs static analysis (via `src.code_analyzer.analyze_source`), and
 - returns a structured result plus a human-readable explanation string.

The module intentionally uses non-opinionated, carefully-worded text for
explanations suitable for an educational debugging assistant.
"""
from typing import Any, Dict, Optional
import os
import joblib
import pandas as pd

from src import code_analyzer
from src.feature_extraction import extract_features
from src import data_loader


def _conservative_unescape(s: Optional[str]) -> str:
    if s is None:
        return ''
    return str(s).replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


def _format_explanation(pred_label: str, prob: Optional[float], static_warnings: list, metrics: dict) -> str:
    parts = []
    # Model-based explanation (careful wording)
    parts.append(f"The model classified this code as **{pred_label}**.")
    if prob is not None:
        parts.append(f"Model confidence: {prob:.2f}.")

    # Static warnings summary
    if static_warnings:
        parts.append(f"Static analysis reported {len(static_warnings)} warning(s). See details below.")
    else:
        parts.append("Static analysis did not find obvious syntax or structural issues.")

    # Add a short metrics summary
    if metrics:
        parts.append(f"Quick metrics: {metrics.get('n_lines', 0)} lines, {metrics.get('n_functions', 0)} function(s).")

    # A gentle note about statistical nature
    parts.append("Note: the model prediction is statistical; absence of static warnings does not guarantee correctness.")
    return " \n".join(parts)


def analyze_and_explain(code: str, model_joblib_path: str = 'models/code_bug_classifier.joblib') -> Dict[str, Any]:
    """Run the full backend analysis (static analysis + ML) and return structured result.

    The returned dict contains:
      - `static`: output of `code_analyzer.analyze_source`
      - `prediction`: {'label': str, 'probability': float or None}
      - `explanation`: human-friendly text

    This function never executes the submitted code.
    """
    result: Dict[str, Any] = {}
    code_proc = _conservative_unescape(code)

    # static analysis
    static = code_analyzer.analyze_source(code_proc)
    result['static'] = static

    # model inference
    try:
        payload = joblib.load(model_joblib_path)
        model = payload.get('model')
        feature_names = payload.get('feature_names')
    except Exception as e:
        result['prediction'] = {'label': None, 'probability': None, 'error': str(e)}
        result['explanation'] = 'Model could not be loaded.'
        return result

    # Extract features using the project's feature extractor
    feats = extract_features(code_proc)

    # Build feature vector in exact saved order
    if feature_names is None:
        raise ValueError('Saved model payload must include feature_names')
    try:
        x = [feats[name] for name in feature_names]
    except KeyError as ke:
        raise KeyError(f'Missing feature for inference: {ke}')

    # Predict
    feature_frame = pd.DataFrame([x], columns=feature_names)
    try:
        pred = model.predict(feature_frame)[0]
    except Exception as e:
        result['prediction'] = {'label': None, 'probability': None, 'error': str(e)}
        result['explanation'] = 'Model prediction failed.'
        return result

    prob = None
    try:
        # many sklearn estimators support predict_proba
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_frame)
            # if binary, take probability for positive class (index 1) when available
            if proba.shape[1] == 2:
                prob = float(proba[0, 1])
            else:
                # fallback: max class probability
                prob = float(proba[0].max())
    except Exception:
        prob = None

    # Map numeric model label to a verified user-facing label.
    def _numeric_label_to_name(numeric_label: int) -> str:
        # Try to infer mapping from dataset metadata in `data/`
        try:
            mapping = data_loader.infer_label_meaning(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        except Exception:
            mapping = None

        if mapping is not None and int(numeric_label) in mapping:
            name = mapping[int(numeric_label)]
        else:
            # Fallback: consult top-level README if present
            try:
                base_readme = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'README.md')
                base_readme = os.path.normpath(base_readme)
                if os.path.exists(base_readme):
                    txt = open(base_readme, 'r', encoding='utf-8', errors='ignore').read().lower()
                    if '0/1 indicate clean vs buggy' in txt or ('0' in txt and 'clean' in txt and '1' in txt and 'bug' in txt):
                        name = {0: 'Clean', 1: 'Buggy'}.get(int(numeric_label), str(numeric_label))
                    else:
                        name = {0: 'Clean', 1: 'Buggy'}.get(int(numeric_label), str(numeric_label))
                else:
                    name = {0: 'Clean', 1: 'Buggy'}.get(int(numeric_label), str(numeric_label))
            except Exception:
                name = {0: 'Clean', 1: 'Buggy'}.get(int(numeric_label), str(numeric_label))

        # Normalize capitalization
        if isinstance(name, str):
            return name.capitalize()
        return str(name)

    numeric = int(pred)
    human_label = _numeric_label_to_name(numeric)

    result['prediction'] = {'numeric_label': numeric, 'label': human_label, 'probability': prob}

    # generate human-friendly explanation
    result['explanation'] = _format_explanation(result['prediction']['label'], prob, static.get('warnings', []), static.get('metrics', {}))
    return result
