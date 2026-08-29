import os
import pandas as pd
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
print(json.dumps({'files': files}, indent=2))

# find first csv
csvs = [f for f in files if f.lower().endswith('.csv')]
if not csvs:
    print('NO_CSV_FOUND')
    raise SystemExit(1)

fname = csvs[0]
path = os.path.join(DATA_DIR, fname)
print(json.dumps({'inspecting_file': fname}, indent=2))

# read with pandas
# try to infer encoding and separators safely
try:
    df = pd.read_csv(path, low_memory=False)
except Exception as e:
    # try with utf-8-sig
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)

out = {}
out['filename'] = fname
out['n_rows'] = int(df.shape[0])
out['n_columns'] = int(df.shape[1])
# columns and dtypes
out['columns'] = [{ 'name': col, 'dtype': str(df[col].dtype) } for col in df.columns.tolist()]
# missing values per column
out['missing_values'] = { col: int(df[col].isna().sum()) for col in df.columns }
# duplicate rows
out['n_duplicate_rows'] = int(df.duplicated().sum())
# check existence of required columns
required_cols = ['code','error_type','explanation','suggested_fix','corrected_code']
out['required_columns_exist'] = { c: (c in df.columns) for c in required_cols }

# error_type distribution
if 'error_type' in df.columns:
    out['error_type_distribution'] = df['error_type'].value_counts(dropna=False).to_dict()
else:
    out['error_type_distribution'] = None

# stage distribution
if 'stage' in df.columns:
    out['stage_distribution'] = df['stage'].value_counts(dropna=False).to_dict()
else:
    out['stage_distribution'] = None

print(json.dumps(out, indent=2))

# show 5 representative records (safe display)
SAMPLE_N = 5
records = []
cols = df.columns.tolist()
for i in range(min(SAMPLE_N, len(df))):
    row = df.iloc[i]
    rec = {}
    for c in cols:
        val = row[c]
        if pd.isna(val):
            rec[c] = None
        else:
                # convert numpy types to Python native types for JSON
                if hasattr(val, 'item'):
                    try:
                        val_conv = val.item()
                    except Exception:
                        val_conv = val
                else:
                    val_conv = val
                if c.lower() in ('code', 'corrected_code', 'snippet'):
                    s = str(val_conv)
                    s = s.replace('\n', '\\n')
                    rec[c] = s[:300] + ('... (truncated)' if len(s) > 300 else '')
                else:
                    rec[c] = val_conv
    records.append(rec)

print('\n5_SAMPLE_RECORDS_JSON_START')
print(json.dumps(records, indent=2, ensure_ascii=False))
print('5_SAMPLE_RECORDS_JSON_END')

# quick data-quality checks
issues = []
# empty code rows
if 'code' in df.columns:
    n_empty_code = int(((df['code'].astype(str).str.strip()=='') | df['code'].isna()).sum())
    if n_empty_code>0:
        issues.append(f'empty_or_missing_code_rows: {n_empty_code}')
# inconsistent labels: check for nulls in error_type
if 'error_type' in df.columns:
    n_null_labels = int(df['error_type'].isna().sum())
    if n_null_labels>0:
        issues.append(f'null_error_type_count: {n_null_labels}')
# duplicate content check (exact duplicates already counted)
if out['n_duplicate_rows']>0:
    issues.append(f'exact_duplicate_rows: {out["n_duplicate_rows"]}')

print('\nDATA_QUALITY_ISSUES_START')
print(json.dumps({'issues': issues}, indent=2))
print('DATA_QUALITY_ISSUES_END')
