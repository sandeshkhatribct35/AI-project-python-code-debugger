"""Data loading utilities for the AI Code Debugger."""
from typing import Tuple
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset CSV with columns: id, language, code, error_type, error_message

    If the file does not exist yet, this function should be updated when dataset is available.
    """
    df = pd.read_csv(path)
    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=random_state)
