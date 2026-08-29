"""Train and compare classifiers on processed features in a leakage-safe way.

Run from project root:
  python scripts/train_models.py

This script does NOT commit or push. It saves model and reports under `models/` and
`reports/` respectively. It uses stratified split, fits scalers only on training data,
and performs cross-validation on the training fold.
"""
from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from src.models import build_classification_pipelines, save_model_joblib
from src.evaluate import evaluate_model


def main(processed_csv: str = 'data/processed_features.csv', out_dir: str = 'reports', model_dir: str = 'models'):
    # Load processed features
    df = pd.read_csv(processed_csv)

    # Verify columns
    if 'label' not in df.columns or 'id' not in df.columns:
        raise SystemExit('Processed CSV must contain `id` and `label` columns')

    # Drop ID and determine feature columns
    id_col = 'id'
    label_col = 'label'
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    X_full = df[feature_cols].copy()
    y = df[label_col].copy()

    # Ensure output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Train/test split (stratified)
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Build pipelines
    pipelines = build_classification_pipelines(random_state=random_state)

    # Cross-validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: List[Dict[str, Any]] = []

    # Important leakage-safety step: detect constant/uninformative features using only X_train
    # (do not use the held-out test set when deciding which features to drop)
    removed: Dict[str, str] = {}
    # Perform detection on a copy of X after splitting below; so re-split here
    # split already performed above; now derive X_train/X_test from X_full
    X_train = X_full.loc[X_train.index].copy()
    X_test = X_full.loc[X_test.index].copy()

    # Remove any feature that is constant in X_train (single unique value)
    for c in list(X_train.columns):
        if X_train[c].nunique(dropna=False) <= 1:
            removed[c] = 'constant_in_train'
            X_train.drop(columns=[c], inplace=True)
            if c in X_test.columns:
                X_test.drop(columns=[c], inplace=True)

    # Specifically check error_msg_len and ensure removal reason recorded
    if 'error_msg_len' in df.columns and 'error_msg_len' not in removed:
        if X_train['error_msg_len'].nunique(dropna=False) <= 1:
            removed['error_msg_len'] = 'constant_in_train'
            if 'error_msg_len' in X_train.columns:
                X_train.drop(columns=['error_msg_len'], inplace=True)
            if 'error_msg_len' in X_test.columns:
                X_test.drop(columns=['error_msg_len'], inplace=True)

    # Final feature columns used by models (order preserved)
    feature_cols_final = list(X_train.columns)

    # For model selection, evaluate cross-validated f1 on training set
    for name, pipeline in pipelines.items():
        # cross-validate on training data only
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1', n_jobs=1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # fit on full training data
        pipeline.fit(X_train, y_train)

        # evaluate on test set
        metrics = evaluate_model(pipeline, X_test, y_test)

        result = {
            'model': name,
            'cv_f1_mean': cv_mean,
            'cv_f1_std': cv_std,
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'confusion_matrix': metrics['confusion_matrix'],
        }
        results.append(result)

        # save confusion matrix as CSV for later inspection
        cm_df = pd.DataFrame(metrics['confusion_matrix'])
        cm_df.to_csv(os.path.join(out_dir, f'confusion_{name}.csv'), index=False)

        # Feature importance / coefficient reporting
        try:
            if name == 'logistic_regression':
                # pipeline: scaler -> clf
                clf = pipeline.named_steps['clf']
                coefs = clf.coef_.ravel()
                coef_df = pd.DataFrame({
                    'feature': feature_cols_final,
                    'coefficient': coefs,
                })
                coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
                coef_df.sort_values('abs_coefficient', ascending=False, inplace=True)
                coef_df.to_csv(os.path.join(out_dir, 'logistic_regression_coefficients.csv'), index=False)
            else:
                clf = pipeline.named_steps['clf']
                importances = getattr(clf, 'feature_importances_', None)
                if importances is not None:
                    imp_df = pd.DataFrame({
                        'feature': feature_cols_final,
                        'importance': importances,
                    })
                    imp_df['abs_importance'] = imp_df['importance'].abs()
                    imp_df.sort_values('abs_importance', ascending=False, inplace=True)
                    fname = f"{name}_feature_importance.csv"
                    imp_df.to_csv(os.path.join(out_dir, fname), index=False)
        except Exception:
            # Do not fail training because of reporting
            pass

    # Choose final model by highest cv_f1_mean
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)

    best_idx = int(results_df['cv_f1_mean'].idxmax())
    best_model_name = results_df.loc[best_idx, 'model']
    best_pipeline = pipelines[best_model_name]

    # Save the selected model and feature names
    model_path = os.path.join(model_dir, 'code_bug_classifier.joblib')
    save_model_joblib(best_pipeline, model_path, feature_names=feature_cols_final)

    # Also write a small JSON summary
    summary = {
        'selected_model': best_model_name,
        'model_path': model_path,
        'removed_features': removed,
        'feature_columns': feature_cols_final,
        'results': results,
    }
    with open(os.path.join(out_dir, 'training_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    # Print brief human-readable training output
    print('Training complete. Summary:')
    print(results_df[['model', 'cv_f1_mean', 'cv_f1_std', 'test_f1']])
    print('\nSelected model:', best_model_name)
    print('Model saved to:', model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_csv', default='data/processed_features.csv')
    parser.add_argument('--out_dir', default='reports')
    parser.add_argument('--model_dir', default='models')
    args = parser.parse_args()
    main(args.processed_csv, args.out_dir, args.model_dir)
