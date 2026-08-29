"""Train models with pair-aware grouping to prevent bug/fix leakage."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from src.models import build_classification_pipelines, save_model_joblib
from src.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'processed_features_augmented.csv')
features = [c for c in df if c not in ('id', 'group_id', 'label', 'error_msg_len')]
X, y, groups = df[features], df['label'], df['group_id']
train_idx, test_idx = next(GroupShuffleSplit(test_size=.2, random_state=42).split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test, train_groups = y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
results, fitted = [], {}
for name, model in build_classification_pipelines(42).items():
    scores = cross_val_score(model, X_train, y_train, groups=train_groups, cv=cv, scoring='f1')
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    results.append({'model': name, 'cv_f1_mean': float(scores.mean()), 'cv_f1_std': float(scores.std()), **{f'test_{k}': v for k, v in metrics.items() if k != 'confusion_matrix'}, 'confusion_matrix': metrics['confusion_matrix']})
    fitted[name] = model
summary = pd.DataFrame(results).sort_values('cv_f1_mean', ascending=False)
best_name = summary.iloc[0]['model']
save_model_joblib(fitted[best_name], ROOT / 'models' / 'code_bug_classifier.joblib', feature_names=features)
summary.to_csv(ROOT / 'reports' / 'model_comparison_augmented.csv', index=False)
(ROOT / 'reports' / 'training_summary_augmented.json').write_text(json.dumps({'selected_model': best_name, 'rows': len(df), 'groups': int(groups.nunique()), 'results': results}, indent=2), encoding='utf-8')
print(summary[['model', 'cv_f1_mean', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']].to_string(index=False))
print('Selected:', best_name)
