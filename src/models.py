"""Model training and selection for error classification."""
from typing import Any
import pickle
from typing import Dict, List

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


def build_classification_pipelines(random_state: int = 42) -> Dict[str, Any]:
    """Return a dict of name -> sklearn estimator (pipelines where appropriate).

    - Logistic Regression: StandardScaler + LogisticRegression
    - Decision Tree: DecisionTreeClassifier (no scaling)
    - Random Forest: RandomForestClassifier (no scaling)
    """
    pipelines: Dict[str, Any] = {}
    pipelines['logistic_regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    pipelines['decision_tree'] = Pipeline([
        ('clf', DecisionTreeClassifier(random_state=random_state)),
    ])
    pipelines['random_forest'] = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ])
    return pipelines


def save_model_joblib(model: Any, path: str, feature_names: List[str] = None) -> None:
    """Save the model and optional metadata using joblib.

    Stores a dict with keys `model` and `feature_names` to ensure feature-order reproducibility.
    """
    payload = {'model': model, 'feature_names': feature_names}
    joblib.dump(payload, path)
