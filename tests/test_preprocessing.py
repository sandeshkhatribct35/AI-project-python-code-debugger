import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import generate_features_df, load_processed_features


def make_sample_csv(path: Path):
    data = [
        {"id": 1, "snippet": "print(1)", "label": 0},
        {"id": 2, "snippet": "def f():\n    return 2", "label": 1},
        {"id": 3, "snippet": "def broken(\n return", "label": 1},
        {"id": 4, "snippet": "", "label": 0},
    ]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def test_generate_features_and_report(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_csv = data_dir / "sample_raw.csv"
    out_csv = data_dir / "processed_features.csv"

    raw_df = make_sample_csv(input_csv)

    processed_df, report = generate_features_df(str(input_csv), str(out_csv))

    # file created and loadable
    assert out_csv.exists()
    loaded = load_processed_features(str(out_csv))
    assert loaded.shape == processed_df.shape

    # id and label preserved
    assert set(loaded['id'].tolist()) == set(raw_df['id'].tolist())
    assert 'label' in loaded.columns

    # some expected features present
    assert 'n_chars' in loaded.columns
    assert 'ast_parse_success' in loaded.columns

    # malformed snippets detected (we had at least one broken snippet)
    assert report['n_malformed_by_ast'] >= 1

    # no destructive modification of the original file (ids and labels preserved,
    # snippets may be read back with NaN for empty strings)
    orig = pd.read_csv(input_csv)
    assert orig['id'].tolist() == raw_df['id'].tolist()
    assert orig['label'].tolist() == raw_df['label'].tolist()
    # compare snippets allowing pandas to read empty as NaN
    orig_snippets = orig['snippet'].fillna("").tolist()
    assert orig_snippets == raw_df['snippet'].astype(str).tolist()


def test_checks_for_missing_and_duplicates(tmp_path: Path):
    data_dir = tmp_path / "data2"
    data_dir.mkdir()
    input_csv = data_dir / "raw2.csv"
    out_csv = data_dir / "processed2.csv"

    # create data with a duplicate id and a constant column scenario
    data = [
        {"id": 1, "snippet": "print(1)", "label": 0},
        {"id": 1, "snippet": "print(2)", "label": 0},
    ]
    pd.DataFrame(data).to_csv(input_csv, index=False)

    processed_df, report = generate_features_df(str(input_csv), str(out_csv))

    # duplicate id detected
    assert report['n_duplicate_ids'] >= 1

    # processed shape matches rows
    assert processed_df.shape[0] == 2
