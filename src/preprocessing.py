"""Preprocessing pipeline: generate feature dataset from raw CSV.

This module provides reproducible, testable functions to read the raw CSV,
apply `extract_features` on each `snippet`, and write out a features CSV.
It also performs basic dataset QA checks (missing, infinite, duplicates,
constant features, class balance) and returns a small report.
"""
from typing import Tuple, Dict, Any, List
import os
import pandas as pd
import numpy as np

from src.feature_extraction import extract_features


def generate_features_df(
    input_csv: str,
    output_csv: str,
    id_col: str = "id",
    code_col: str = "snippet",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read `input_csv`, extract features for each snippet and save to `output_csv`.

    Returns (processed_df, report).

    The original CSV is not modified.
    """
    df = pd.read_csv(input_csv, low_memory=False)

    # Ensure expected columns
    if id_col not in df.columns or code_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Input CSV must contain columns: {id_col}, {code_col}, {label_col}")

    rows: List[Dict[str, Any]] = []
    malformed_count = 0
    unescape_warnings: int = 0

    def conservative_unescape(s: str) -> str:
        # Only replace literal two-character escape sequences with real
        # characters. Do not apply broad unicode_escape decoding.
        if s is None:
            return ''
        # operate on str
        out = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        return out
    for _, row in df.iterrows():
        sample_id = row[id_col]
        code = row.get(code_col, "")
        err = row.get("error_message", None)
        # Work on an in-memory transformed copy of the snippet; keep raw CSV unchanged
        code_proc = conservative_unescape(str(code))
        try:
            feats = extract_features(code_proc, err)
        except Exception:
            # Defensive: if feature extraction fails entirely, create a zero-filled feature dict
            feats = {
                'n_chars': 0,
                'n_lines': 0,
                'n_tokens': 0,
                'ast_parse_success': 0,
                'ast_syntax_error': 1,
            }
        # Track whether unescaping changed the snippet (for diagnostics)
        if str(code).find('\\n') != -1 or str(code).find('\\t') != -1 or str(code).find('\\r') != -1:
            # increment a simple counter (not used elsewhere programmatically)
            unescape_warnings += 1
        # track malformed by AST flag
        if feats.get('ast_parse_success', 0) == 0:
            malformed_count += 1
        # do not use label to compute features — only attach it later
        row_out = {id_col: sample_id}
        row_out.update({f: v for f, v in feats.items()})
        row_out[label_col] = row[label_col]
        rows.append(row_out)

    processed_df = pd.DataFrame(rows)

    # Basic QA checks
    report: Dict[str, Any] = {}
    report['final_shape'] = processed_df.shape
    report['feature_names'] = [c for c in processed_df.columns if c not in [id_col, label_col]]
    report['label_distribution'] = processed_df[label_col].value_counts(dropna=False).to_dict()
    report['missing_values'] = processed_df.isnull().sum().to_dict()

    # Infinite / non-finite values (use NumPy to avoid DataFrame applymap issues)
    numeric = processed_df.select_dtypes(include=[np.number])
    if not numeric.empty:
        try:
            vals = numeric.to_numpy(dtype=float)
            nonfinite_mask = ~np.isfinite(vals)
            report['n_infinite_values'] = int(np.count_nonzero(nonfinite_mask))
        except Exception:
            # fallback conservative value
            report['n_infinite_values'] = -1
    else:
        report['n_infinite_values'] = 0

    # Duplicate rows (based on id)
    report['n_duplicate_ids'] = int(processed_df.duplicated(subset=[id_col]).sum())

    # Constant features (columns with single unique value)
    constant_features = [c for c in report['feature_names'] if processed_df[c].nunique(dropna=False) <= 1]
    report['constant_features'] = constant_features

    # Class imbalance basic: counts and ratio
    counts = processed_df[label_col].value_counts()
    if len(counts) > 0:
        report['class_counts'] = counts.to_dict()
        report['majority_fraction'] = float(counts.max() / counts.sum())
    else:
        report['class_counts'] = {}
        report['majority_fraction'] = None

    report['n_malformed_by_ast'] = int(malformed_count)
    report['n_snippets_with_literal_escapes'] = int(unescape_warnings)

    # basic statistics for numeric features
    report['numeric_describe'] = numeric.describe().to_dict()

    # Save processed CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)

    return processed_df, report


def load_processed_features(path: str) -> pd.DataFrame:
    """Load a processed features CSV previously generated."""
    return pd.read_csv(path)
