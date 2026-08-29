"""Streamlit interface for the safe Python Code Debugger."""
from pathlib import Path
import json
import streamlit as st

from src.explainers import analyze_and_explain
from src.recommendations import recommendations_from_warnings, unique_recommendations

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
SAMPLES = {
    "Choose a sample": "",
    "Valid function": "def calculate_total(values):\n    return sum(values)\n\nprint(calculate_total([10, 20, 30]))",
    "Syntax error": "def calculate_average(numbers)\n    total = sum(numbers)\n    return total / len(numbers)",
    "Delimiter mismatch": "numbers = [10, 20, 30]\nprint(sum(numbers)",
    "Risky dynamic execution": "user_expression = '2 + 2'\nresult = eval(user_expression)\nprint(result)",
}

st.set_page_config(page_title="AI Code Debugger", page_icon="🐍", layout="wide")

@st.cache_data
def load_training_summary():
    path = REPORTS / "training_summary_augmented.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

if "code_input" not in st.session_state:
    st.session_state.code_input = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None

with st.sidebar:
    st.title("🐍 Code Debugger")
    st.caption("Safe static analysis + ML classification")
    st.divider()
    sample = st.selectbox("Try a safe sample", list(SAMPLES))
    if st.button("Load sample", use_container_width=True) and sample != "Choose a sample":
        st.session_state.code_input = SAMPLES[sample]
        st.session_state.analysis = None
    st.divider()
    st.subheader("Safety promise")
    st.write("Your code is never run, imported, or modified. The app only parses text and extracts static features.")
    st.caption("Python is currently the supported language.")

st.title("AI-Based Python Code Debugger")
st.caption("Analyze Python code safely, inspect static findings, and view an experimental Clean/Buggy classification.")

analyze_tab, performance_tab, about_tab = st.tabs(["🔎 Analyze code", "📊 Model performance", "ℹ️ About"])

with analyze_tab:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("1. Paste Python code")
        st.text_area(
            "Source code",
            key="code_input",
            height=360,
            placeholder="def divide(a, b):\n    return a / b",
            max_chars=50_000,
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("Analyze safely", type="primary", use_container_width=True)
    with right:
        st.subheader("What you will receive")
        st.markdown("- Clean or Buggy classification\n- Model confidence (not a real-world probability)\n- Syntax and static findings\n- Conservative next steps")
        st.info("For the strongest evidence, use static findings and tests. The ML result is a statistical signal.")

    if analyze_clicked:
        if not st.session_state.code_input.strip():
            st.warning("Paste Python code or load a sample first.")
        else:
            with st.spinner("Analyzing without executing your code..."):
                st.session_state.analysis = analyze_and_explain(st.session_state.code_input)

    result = st.session_state.analysis
    if result:
        static = result["static"]
        prediction = result["prediction"]
        st.divider()
        st.subheader("2. Analysis result")
        if prediction.get("error"):
            st.error("Static analysis completed, but the trained model was unavailable.")
            st.caption(prediction["error"])
        else:
            label = prediction["label"]
            confidence = prediction.get("probability")
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Classification", label)
            metric2.metric("Model confidence", f"{confidence:.1%}" if confidence is not None else "Unavailable")
            metric3.metric("Static findings", len(static.get("warnings", [])))
            if confidence is not None and 0.40 < confidence < 0.60:
                st.info("The model is uncertain. Prioritize the static findings and run focused tests.")

        syntax = static.get("syntax_error")
        if static.get("parse_success"):
            st.success("Python syntax parsed successfully.")
        else:
            line = syntax.get("line") if syntax else None
            location = f" on line {line}" if line else ""
            st.error(f"Python syntax could not be parsed{location}: {syntax.get('message', 'unknown error') if syntax else 'unknown error'}")

        findings_tab, guidance_tab, metrics_tab = st.tabs(["Findings", "Explanation & next steps", "Static metrics"])
        warnings = static.get("warnings", [])
        with findings_tab:
            if not warnings:
                st.success("No obvious static issues were found.")
            for warning in warnings:
                severity = "Error" if warning.get("severity") == "error" else "Warning"
                line = warning.get("line")
                title = f"{severity}: {warning['message']}" + (f" — line {line}" if line else "")
                with st.expander(title, expanded=warning.get("severity") == "error"):
                    st.write(warning.get("explanation", ""))
                    if line and 1 <= line <= len(st.session_state.code_input.splitlines()):
                        source = st.session_state.code_input.splitlines()
                        start, end = max(0, line - 2), min(len(source), line + 1)
                        context = "\n".join(f"{number + 1:>3} | {source[number]}" for number in range(start, end))
                        st.code(context, language="python")
        with guidance_tab:
            st.markdown(result["explanation"])
            recommendations = unique_recommendations(recommendations_from_warnings(warnings))
            st.markdown("#### Recommended next steps")
            if recommendations:
                for recommendation in recommendations:
                    st.markdown(f"- {recommendation['suggestion']}")
            else:
                st.markdown("- Add focused unit tests for the code path you are changing.\n- Use a debugger or logging to inspect runtime values.")
            st.warning("AI suggestions are recommendations. Verify them before changing your code.")
        with metrics_tab:
            st.json(static.get("metrics", {}))

with performance_tab:
    summary = load_training_summary()
    st.subheader("Leakage-safe grouped evaluation")
    st.caption("Buggy/fixed pairs were kept together during splitting, preventing one version from leaking into the other split.")
    if summary:
        selected = next(item for item in summary["results"] if item["model"] == summary["selected_model"])
        cols = st.columns(4)
        cols[0].metric("Selected model", summary["selected_model"].replace("_", " ").title())
        cols[1].metric("Test accuracy", f"{selected['test_accuracy']:.1%}")
        cols[2].metric("Test F1", f"{selected['test_f1']:.1%}")
        cols[3].metric("CV F1", f"{selected['cv_f1_mean']:.1%}")
    chart1, chart2 = st.columns(2)
    with chart1:
        image = REPORTS / "model_performance_comparison.png"
        if image.exists(): st.image(str(image), caption="Model comparison")
    with chart2:
        image = REPORTS / "best_model_confusion_matrix.png"
        if image.exists(): st.image(str(image), caption="Best model confusion matrix")
    image = REPORTS / "augmented_dataset_label_distribution.png"
    if image.exists(): st.image(str(image), caption="Training label distribution", width=450)

with about_tab:
    st.subheader("How the debugger works")
    st.markdown("1. The app parses code with Python AST and token tools.\n2. It extracts static features without running your code.\n3. The trained classifier returns Clean or Buggy plus a confidence score.\n4. Static findings and recommendations are displayed for review.")
    st.subheader("Limitations")
    st.write("The classifier does not understand every program’s full meaning. A clean result does not guarantee correctness, and static analysis cannot detect every runtime or business-logic error.")
