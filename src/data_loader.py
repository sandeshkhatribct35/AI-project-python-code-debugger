"""Data loading utilities for the AI Code Debugger."""
from typing import Tuple
import pandas as pd
import os
from typing import Optional, Dict


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset CSV with columns: id, language, code, error_type, error_message

    If the file does not exist yet, this function should be updated when dataset is available.
    """
    df = pd.read_csv(path, low_memory=False)
    return df


def infer_label_meaning(data_dir: str) -> Optional[Dict[int, str]]:
    """Attempt to infer label meaning (e.g., 0=clean,1=buggy) from local README or metadata.

    Returns a mapping {label_value: meaning} when found, otherwise None.
    This function does not assume semantics; it searches for hints in files under `data_dir`.
    """
    # look for README or README_dataset files in the data directory
    candidates = [
        os.path.join(data_dir, 'README_dataset.md'),
        os.path.join(data_dir, 'README.md'),
        os.path.join(data_dir, 'readme.md'),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                text = open(c, 'r', encoding='utf-8', errors='ignore').read().lower()
                if '0' in text and 'clean' in text and '1' in text and 'bug' in text:
                    return {0: 'clean', 1: 'buggy'}
            except Exception:
                continue
    # no explicit mapping found
    return None


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=random_state)
