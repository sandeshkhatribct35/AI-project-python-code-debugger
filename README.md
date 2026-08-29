# AI-Based Intelligent Code Debugging and Error Explanation System

Mini project for "AI Applications and Ethics". The app accepts Python source
code, performs static analysis without executing it, and combines those findings
with a trained binary classifier that estimates whether a snippet resembles the
project's buggy training examples.

## Quick start

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app shows:

- a clean/buggy model classification and estimated bug probability;
- Python syntax status and source line when parsing fails;
- static findings for delimiter, indentation, bare-except, and `eval`/`exec`
  patterns;
- conservative next-step recommendations and inspectable code metrics.

The submitted code is never run by the application.

## Train the model

After preprocessing the dataset into `data/processed_features.csv`, run:

```bash
python scripts/train_models.py
```

This writes the selected model to `models/code_bug_classifier.joblib` and
evaluation outputs to `reports/`. Run the automated checks with:

```bash
python -m pytest -q
```

## Current evaluation and limitations

`reports/training_summary.json` records the current results. The selected
logistic-regression model has mean cross-validation F1 of about **0.51** and a
held-out test F1 of about **0.56**. This is close to a weak baseline, so its
output must be treated as an educational risk signal rather than proof that code
is correct or defective. Static-analysis findings are also heuristic and do not
detect runtime failures such as incorrect business logic or external-service
errors.

See `notebooks/` for EDA and `src/` for core modules.

Dataset note
------------

The primary dataset placed in `data/` is a small, synthetic Python snippet dataset (`dataset_realistic_bug.csv`) containing binary labels (`label`) where 0/1 indicate clean vs buggy snippets (confirm mapping in dataset README). This CSV does NOT contain multi-class `error_type`, human-written `explanation`, `suggested_fix`, or `corrected_code` fields described by some Kaggle pages. Because of this, the initial ML task will be a binary buggy-vs-clean classifier unless additional labeled files are provided or sourced.

This limitation is intentional and will be discussed in the project's Ethics section. If you prefer to target multi-class error classification, we'll need a different dataset with explicit error-type labels or additional annotation steps.
