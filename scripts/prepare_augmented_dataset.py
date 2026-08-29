"""Build a leakage-safe augmented training set from the Kaggle bug/fix pairs."""
from pathlib import Path
import pandas as pd
from src.feature_extraction import extract_features

ROOT = Path(__file__).resolve().parents[1]
source = pd.read_csv(ROOT / 'data' / 'dataset_realistic_bug.csv')
pairs = pd.read_csv(ROOT / 'data' / 'python_code_bug_and_fix_pairs' / 'code_bug_fix_pairs.csv')

original = pd.DataFrame({'id': 'original-' + source['id'].astype(str),
                         'group_id': 'original-' + source['id'].astype(str),
                         'snippet': source['snippet'].fillna(''), 'label': source['label'].astype(int)})
buggy = pd.DataFrame({'id': 'kaggle-' + pairs['id'].astype(str) + '-buggy',
                      'group_id': 'kaggle-' + pairs['id'].astype(str),
                      'snippet': pairs['buggy_code'].fillna(''), 'label': 1})
fixed = pd.DataFrame({'id': 'kaggle-' + pairs['id'].astype(str) + '-fixed',
                      'group_id': 'kaggle-' + pairs['id'].astype(str),
                      'snippet': pairs['fixed_code'].fillna(''), 'label': 0})
raw = pd.concat([original, buggy, fixed], ignore_index=True).drop_duplicates(subset=['snippet', 'label']).reset_index(drop=True)
features = pd.DataFrame([extract_features(code) for code in raw['snippet']])
out = pd.concat([raw[['id', 'group_id', 'label']], features], axis=1)
out.to_csv(ROOT / 'data' / 'processed_features_augmented.csv', index=False)
print({'rows': len(out), 'groups': out['group_id'].nunique(), 'labels': out['label'].value_counts().to_dict()})
