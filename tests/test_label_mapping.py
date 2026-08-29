import os
import json
from src.explainers import analyze_and_explain
from src import data_loader


def _derive_project_label_mapping():
    # Try dataset-level README under data/
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    mapping = data_loader.infer_label_meaning(data_dir)
    if mapping:
        return {int(k): v.capitalize() for k, v in mapping.items()}

    # Fallback: parse top-level README.md for an explicit sentence like
    # "0/1 indicate clean vs buggy"
    base_readme = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md'))
    if os.path.exists(base_readme):
        txt = open(base_readme, 'r', encoding='utf-8', errors='ignore').read().lower()
        if '0/1 indicate clean vs buggy' in txt or ('0' in txt and 'clean' in txt and '1' in txt and 'bug' in txt):
            return {0: 'Clean', 1: 'Buggy'}

    # If we cannot derive mapping programmatically, fail the test intentionally
    return None


def test_project_label_mapping_applied_in_explainers():
    mapping = _derive_project_label_mapping()
    assert mapping is not None, 'Could not derive project label mapping from data/ or README.md'

    # Call analyze_and_explain and verify the returned human label matches the derived mapping
    res = analyze_and_explain("def foo():\n    return 1\n")
    assert 'prediction' in res
    pred = res['prediction']
    assert 'numeric_label' in pred and 'label' in pred
    numeric = int(pred['numeric_label'])
    human = pred['label']
    assert numeric in mapping, f'Numeric label {numeric} not in derived mapping'
    assert human == mapping[numeric], f"Explainer label '{human}' does not match derived mapping {mapping[numeric]} for numeric {numeric}"
