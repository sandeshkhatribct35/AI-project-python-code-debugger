# AI-Based Intelligent Code Debugging and Error Explanation System

Mini project for "AI Applications and Ethics"

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

See `notebooks/` for EDA and `src/` for core modules.

Dataset note
------------

The primary dataset placed in `data/` is a small, synthetic Python snippet dataset (`dataset_realistic_bug.csv`) containing binary labels (`label`) where 0/1 indicate clean vs buggy snippets (confirm mapping in dataset README). This CSV does NOT contain multi-class `error_type`, human-written `explanation`, `suggested_fix`, or `corrected_code` fields described by some Kaggle pages. Because of this, the initial ML task will be a binary buggy-vs-clean classifier unless additional labeled files are provided or sourced.

This limitation is intentional and will be discussed in the project's Ethics section. If you prefer to target multi-class error classification, we'll need a different dataset with explicit error-type labels or additional annotation steps.
