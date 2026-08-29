import pandas as pd
import os

data_path = os.path.join('data', 'dataset_realistic_bug.csv')
print('FILE:', data_path)

df = pd.read_csv(data_path)
rows, cols = df.shape
print('ROWS:', rows)
print('COLS:', cols)

print('\nCOLUMNS AND DTYPES:')
print(df.dtypes)

print('\nMISSING VALUES PER COLUMN:')
missing = df.isnull().sum()
print(missing)

print('\nDUPLICATE ROWS:', df.duplicated().sum())

# error_type distribution
if 'error_type' in df.columns:
    print('\nUNIQUE error_type VALUES AND COUNTS:')
    print(df['error_type'].value_counts(dropna=False))
else:
    print('\nCOLUMN error_type NOT FOUND')

# stage distribution
if 'stage' in df.columns:
    print('\nUNIQUE stage VALUES AND COUNTS:')
    print(df['stage'].value_counts(dropna=False))
else:
    print('\nCOLUMN stage NOT FOUND')

# label distribution (present in this CSV)
if 'label' in df.columns:
    print('\nLABEL column VALUE COUNTS:')
    print(df['label'].value_counts(dropna=False))

# Check for presence of required columns
required = ['code','error_type','explanation','suggested_fix','corrected_code']
print('\nREQUIRED COLUMNS PRESENCE:')
for c in required:
    print(c, ':', c in df.columns)

# Show 5 representative records (mask code field if present)
print('\n5 REPRESENTATIVE RECORDS:')
preview = df.head(5).copy()
if 'code' in preview.columns:
    def mask_code(s):
        if pd.isna(s):
            return s
        s2 = str(s)
        s2 = s2.replace('\n','\\n')
        if len(s2) > 200:
            return s2[:200] + '...[truncated]'
        return s2
    preview['code_preview'] = preview['code'].apply(mask_code)
    # drop full code from display
    preview = preview.drop(columns=['code'])
print(preview.to_string(index=False))

# Basic data quality checks
print('\nBASIC DATA QUALITY CHECKS:')
# empty code rows
if 'code' in df.columns:
    empty_code = df['code'].astype(str).str.strip().replace({'nan': ''}).eq('').sum()
    print('Empty or blank `code` rows:', empty_code)
# null error_type
if 'error_type' in df.columns:
    null_errors = df['error_type'].isnull().sum()
    print('Null `error_type` rows:', null_errors)

# inconsistent labels heuristic: leading/trailing spaces
if 'error_type' in df.columns:
    stripped = df['error_type'].astype(str).str.strip()
    n_diff = (stripped != df['error_type'].astype(str)).sum()
    print('error_type values with leading/trailing whitespace:', n_diff)

print('\nINSPECTION COMPLETE')
