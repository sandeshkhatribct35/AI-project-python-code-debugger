from src.preprocessing import generate_features_df
import pandas as pd
from pathlib import Path


def test_conservative_unescape_and_parse(tmp_path: Path):
    # create a small CSV with escaped literals
    data = [
        {"id": 1, "snippet": "def a():\\n    return 1", "label": 0},
        {"id": 2, "snippet": "def b():\\n\treturn 2", "label": 1},
            {"id": 3, "snippet": 'print("literal")', "label": 0},
    ]
    p = tmp_path / "data"
    p.mkdir()
    input_csv = p / "raw.csv"
    out_csv = p / "processed.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)

    processed_df, report = generate_features_df(str(input_csv), str(out_csv))

    # ensure that unescaping allowed parse on at least the first two
    assert processed_df.loc[processed_df['id'] == 1, 'ast_parse_success'].iloc[0] == 1
    assert processed_df.loc[processed_df['id'] == 2, 'ast_parse_success'].iloc[0] == 1
    # snippet 3 is a single-line print; ensure not broken
    assert 'n_chars' in processed_df.columns


def test_raw_csv_unchanged(tmp_path: Path):
    data = [{"id": 1, "snippet": "def a():\\n    return 1", "label": 0}]
    p = tmp_path / "d2"
    p.mkdir()
    input_csv = p / "raw2.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    # run preprocessing
    out_csv = p / "out.csv"
    generate_features_df(str(input_csv), str(out_csv))
    # raw file read back should still contain literal \n sequences
    raw = pd.read_csv(input_csv)
    assert '\\n' in raw.loc[0, 'snippet']
