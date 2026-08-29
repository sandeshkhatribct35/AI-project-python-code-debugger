import ast
import json
import pandas as pd
import numpy as np
from typing import Any, Dict


def inspect_snippet(code: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info['repr'] = repr(code)
    info['len'] = len(code)
    info['contains_newline_char'] = '\\n' in code
    info['contains_actual_newline'] = '\n' in code
    try:
        ast.parse(code)
        info['ast_ok'] = True
        info['exception'] = None
    except SyntaxError as e:
        info['ast_ok'] = False
        info['exception'] = {
            'type': 'SyntaxError',
            'msg': str(e),
            'lineno': e.lineno,
            'offset': e.offset,
            'text': e.text,
        }
    except Exception as e:
        info['ast_ok'] = False
        info['exception'] = {
            'type': type(e).__name__,
            'msg': str(e),
        }
    return info


def main():
    raw_path = 'data/dataset_realistic_bug.csv'
    proc_path = 'data/processed_features.csv'

    df = pd.read_csv(raw_path, low_memory=False)

    # select up to 5 from each label present
    samples = []
    labels = df['label'].unique().tolist()
    for lbl in labels:
        sub = df[df['label'] == lbl]
        take = min(5, len(sub))
        for _, r in sub.head(take).iterrows():
            samples.append({'id': r['id'], 'label': r['label'], 'snippet': r['snippet']})

    inspected = []
    for s in samples:
        info = inspect_snippet(str(s['snippet']))
        info['id'] = int(s['id'])
        info['label'] = int(s['label'])
        inspected.append(info)

    # now overall QA on processed file
    proc = pd.read_csv(proc_path, low_memory=False)
    report: Dict[str, Any] = {}
    report['processed_shape'] = proc.shape
    report['label_distribution'] = proc['label'].value_counts(dropna=False).to_dict()
    report['missing_values'] = proc.isnull().sum().to_dict()
    # non-finite
    numeric = proc.select_dtypes(include=[np.number])
    try:
        vals = numeric.to_numpy(dtype=float)
        report['n_nonfinite'] = int((~np.isfinite(vals)).sum())
    except Exception:
        report['n_nonfinite'] = None
    # duplicate rows
    report['n_duplicate_rows'] = int(proc.duplicated().sum())
    # duplicate IDs
    report['n_duplicate_ids'] = int(proc.duplicated(subset=['id']).sum())
    # constant features
    feature_cols = [c for c in proc.columns if c not in ['id', 'label']]
    constant = [c for c in feature_cols if proc[c].nunique(dropna=False) <= 1]
    report['constant_features'] = constant

    out = {'samples': inspected, 'report': report}
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
