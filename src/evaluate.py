"""Model evaluation utilities."""
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import Dict, Any


def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """Evaluate a fitted model on test data and return metrics and confusion matrix.

    The confusion matrix is returned as a nested list for easy serialization.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_predictions(y_true, y_pred) -> Dict[str, Any]:
    """Evaluate predictions provided directly (useful for cross-validated estimates)."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }
