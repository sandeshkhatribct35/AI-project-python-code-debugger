"""Model evaluation utilities."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}
