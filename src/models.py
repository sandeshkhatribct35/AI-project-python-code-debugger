"""Model training and selection for error classification."""
from typing import Any
import pickle


def train_model(X, y, model_type: str = 'random_forest') -> Any:
    """Train and return a fitted model. Replace with real training code when dataset available."""
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
