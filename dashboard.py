"""
THESIS PIPELINE DASHBOARD
==========================
Interactive Streamlit dashboard that runs the full EDA + Hypothesis pipeline
and reports results at each step using a tab-based layout.

Usage:
    python -m streamlit run dashboard.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
OUTPUT_ROOT = BASE_DIR / "outputs"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def run_command(cmd, label=""):
    """Run a subprocess command and return (success, stdout, stderr, duration)."""
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    duration = time.time() - start
    return result.returncode == 0, result.stdout, result.stderr, duration


def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def generate_fallback_summary(validation_dir, summary_dir, dataset_name):
    """Generate a structured summary from validation results without needing LLM."""
    val_results_path = validation_dir / f"{dataset_name}_train_validation_results.csv"
    df = load_csv_safe(val_results_path)

    if df is None or df.empty:
        st.info("No validation results to summarize.")
        return

    accepted = df[df["final_accepted"] == True] if "final_accepted" in df.columns else pd.DataFrame()

    lines = [f"# {dataset_name} — Pipeline Summary\n"]
    lines.append("## Overview\n")
    lines.append(f"- **Total hypotheses tested**: {len(df)}")
    lines.append(f"- **Final accepted**: {len(accepted)}")
    if len(df) > 0:
        lines.append(f"- **Acceptance rate**: {len(accepted)/len(df)*100:.1f}%\n")

    if not accepted.empty:
        lines.append("## Accepted Hypotheses\n")
        for _, row in accepted.iterrows():
            h_id = row.get("id", "")
            claim = row.get("claim", "N/A")
            effect = row.get("effect", float("nan"))
            p_val = row.get("p_value", float("nan"))
            test = row.get("primary_test", "")
            direction = row.get("direction_expected", "")
            lines.append(f"### {h_id}")
            lines.append(f"**Claim**: {claim}\n")
            lines.append(f"- Test: `{test}` | Effect: `{effect:.4f}` | p-value: `{p_val:.4e}` | Direction: {direction}\n")
    else:
        lines.append("## Result\n")
        lines.append("No hypotheses passed all validation criteria.\n")
        if "validation_status" in df.columns:
            lines.append("### Failure reasons:\n")
            for status, count in df["validation_status"].value_counts().items():
                lines.append(f"- {status}: {count}")

    summary_text = "\n".join(lines)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"{dataset_name}_final_summary.md"
    summary_file.write_text(summary_text, encoding="utf-8")
    st.markdown(summary_text)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Thesis Pipeline Dashboard",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(":material/science: Reducing False Discoveries in EDA with Structured AI Reasoning")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(":material/folder_open: Data Input")

    input_mode = st.radio("Choose input method:", ["Upload CSV", "Select existing dataset"])

    input_csv_path = None
    dataset_name = None

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            data_dir = BASE_DIR / "DATA"
            data_dir.mkdir(exist_ok=True)
            save_path = data_dir / uploaded_file.name
            save_path.write_bytes(uploaded_file.getvalue())
            input_csv_path = save_path
            dataset_name = save_path.stem
            st.success(f"Loaded: {uploaded_file.name}")
    else:
        data_dir = BASE_DIR / "DATA"
        csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
        if csv_files:
            selected = st.selectbox("Select dataset:", csv_files, format_func=lambda x: x.stem)
            input_csv_path = selected
            dataset_name = selected.stem
        else:
            st.warning("No CSV files found in DATA folder.")

    st.divider()
    st.header(":material/tune: Parameters")
    stratify_col = st.text_input("Stratify column (optional)", value="")
    test_size = st.slider("Test split ratio", 0.1, 0.5, 0.3, 0.05)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    min_abs_r = st.slider("Min |r| (num-num)", 0.05, 0.50, 0.15, 0.05)
    min_cramers_v = st.slider("Min Cramér's V (cat-cat)", 0.05, 0.50, 0.20, 0.05)
    min_eta2 = st.slider("Min η² (num-cat)", 0.01, 0.20, 0.05, 0.01)
    bootstrap_iters = st.slider("Bootstrap iterations", 50, 500, 200, 50,
                                help="More = more stable but slower.")

    st.divider()
    skip_llm = st.checkbox("Skip LLM steps (6 & 8)", value=False,
                           help="Skip if Ollama is not available")

    run_pipeline = st.button(":material/rocket_launch: Run Pipeline", type="primary", use_container_width=True)


# ============================================================
# PIPELINE EXECUTION (stores results in session_state)
# ============================================================

def execute_pipeline(input_csv_path, dataset_name):
    """Run full pipeline and store results in st.session_state."""
    results = {"status": {}, "errors": {}}

    dataset_root = OUTPUT_ROOT / dataset_name
    data_dir = dataset_root / "data"
    preprocess_dir = dataset_root / "preprocess"
    associations_dir = dataset_root / "associations"
    prompt_dir = dataset_root / "prompt"
    hypotheses_dir = dataset_root / "hypotheses"
    validation_dir = dataset_root / "validation"
    summary_dir = dataset_root / "summary"

    for d in [data_dir, preprocess_dir, associations_dir, prompt_dir, hypotheses_dir, validation_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results["paths"] = {
        "dataset_root": dataset_root,
        "data_dir": data_dir,
        "preprocess_dir": preprocess_dir,
        "associations_dir": associations_dir,
        "prompt_dir": prompt_dir,
        "hypotheses_dir": hypotheses_dir,
        "validation_dir": validation_dir,
        "summary_dir": summary_dir,
    }

    progress = st.progress(0, text="Starting pipeline...")

    # Step 1: Split
    progress.progress(5, text="Step 1: Splitting data...")
    cmd = [PYTHON_EXE, str(BASE_DIR / "split_data.py"),
           "--input", str(input_csv_path), "--output-dir", str(data_dir), "--test-size", str(test_size)]
    if stratify_col:
        cmd.extend(["--stratify-col", stratify_col])
    ok, out, err, dur = run_command(cmd)
    results["status"]["step1"] = ok
    results["step1_duration"] = dur
    if not ok:
        results["errors"]["step1"] = err
        progress.progress(100, text="Pipeline failed at Step 1.")
        return results

    # Step 2: Preprocess train
    progress.progress(15, text="Step 2: Preprocessing train...")
    train_csv = data_dir / f"{dataset_name}_train.csv"
    cmd = [PYTHON_EXE, str(BASE_DIR / "preprocess.py"),
           "--input", str(train_csv), "--output-dir", str(preprocess_dir), "--mode", "fit_transform"]
    ok, out, err, dur = run_command(cmd)
    results["status"]["step2"] = ok
    results["step2_duration"] = dur
    if not ok:
        results["errors"]["step2"] = err
        progress.progress(100, text="Pipeline failed at Step 2.")
        return results

    # Step 3: Preprocess test
    progress.progress(25, text="Step 3: Preprocessing test...")
    test_csv = data_dir / f"{dataset_name}_test.csv"
    state_path = preprocess_dir / f"{dataset_name}_train_preprocessing_state.json"
    cmd = [PYTHON_EXE, str(BASE_DIR / "preprocess.py"),
           "--input", str(test_csv), "--output-dir", str(preprocess_dir),
           "--mode", "transform_only", "--state-json", str(state_path)]
    ok, out, err, dur = run_command(cmd)
    results["status"]["step3"] = ok
    results["step3_duration"] = dur
    if not ok:
        results["errors"]["step3"] = err
        progress.progress(100, text="Pipeline failed at Step 3.")
        return results

    # Step 4: Associations
    progress.progress(35, text="Step 4: Computing associations...")
    train_cleaned = preprocess_dir / f"{dataset_name}_train_cleaned.csv"
    train_profile = preprocess_dir / f"{dataset_name}_train_profile.json"
    cmd = [PYTHON_EXE, str(BASE_DIR / "run_all_associations.py"),
           "--input-csv", str(train_cleaned), "--profile-json", str(train_profile),
           "--output-dir", str(associations_dir), "--alpha", str(alpha),
           "--min-abs-r", str(min_abs_r), "--min-cramers-v", str(min_cramers_v),
           "--min-eta2", str(min_eta2), "--min-rank-biserial", "0.15"]
    ok, out, err, dur = run_command(cmd)
    results["status"]["step4"] = ok
    results["step4_duration"] = dur
    if not ok:
        results["errors"]["step4"] = err
        progress.progress(100, text="Pipeline failed at Step 4.")
        return results

    # Step 5: Prompt generation
    progress.progress(50, text="Step 5: Building evidence pack...")
    filtered_path = associations_dir / f"{dataset_name}_train_associations_filtered.csv"
    cmd = [PYTHON_EXE, str(BASE_DIR / "prompt_generate.py"),
           "--profile-json", str(train_profile), "--filtered-assoc-csv", str(filtered_path),
           "--output-dir", str(prompt_dir)]
    ok, out, err, dur = run_command(cmd)
    results["status"]["step5"] = ok
    results["step5_duration"] = dur
    if not ok:
        results["errors"]["step5"] = err
        progress.progress(100, text="Pipeline failed at Step 5.")
        return results

    # Step 6: LLM Hypothesis Generation
    progress.progress(60, text="Step 6: Generating hypotheses...")
    hypotheses_json_path = hypotheses_dir / f"{dataset_name}_train_hypotheses.json"
    if skip_llm:
        results["status"]["step6"] = hypotheses_json_path.exists()
        results["step6_skipped"] = True
    else:
        prompt_file = prompt_dir / f"{dataset_name}_train_hypothesis_prompt.txt"
        cmd = [PYTHON_EXE, str(BASE_DIR / "generate_hypotheses.py"),
               "--prompt-file", str(prompt_file), "--output-dir", str(hypotheses_dir)]
        ok, out, err, dur = run_command(cmd)
        results["status"]["step6"] = ok or hypotheses_json_path.exists()
        results["step6_duration"] = dur
        if not ok:
            results["errors"]["step6"] = err

    if not hypotheses_json_path.exists():
        progress.progress(100, text="Pipeline stopped — no hypotheses available.")
        return results

    # Step 7: Validation + Bootstrap
    progress.progress(75, text=f"Step 7: Validating ({bootstrap_iters} bootstrap iters)...")
    test_cleaned_path = preprocess_dir / f"{dataset_name}_test_cleaned.csv"
    cmd = [PYTHON_EXE, str(BASE_DIR / "validate_hypotheses.py"),
           "--hypotheses-json", str(hypotheses_json_path), "--test-csv", str(test_cleaned_path),
           "--output-dir", str(validation_dir), "--alpha", str(alpha),
           "--min-abs-r", str(min_abs_r), "--min-cramers-v", str(min_cramers_v),
           "--min-eta2", str(min_eta2), "--min-rank-biserial", "0.15",
           "--bootstrap-iterations", str(bootstrap_iters)]
    ok, out, err, dur = run_command(cmd)
    results["status"]["step7"] = ok
    results["step7_duration"] = dur
    if not ok:
        results["errors"]["step7"] = err
        progress.progress(100, text="Pipeline failed at Step 7.")
        return results

    # Step 8: Summary
    progress.progress(90, text="Step 8: Generating summary...")
    validation_results_csv = validation_dir / f"{dataset_name}_train_validation_results.csv"
    if skip_llm:
        results["status"]["step8"] = True
        results["step8_skipped"] = True
    else:
        cmd = [PYTHON_EXE, str(BASE_DIR / "summarize_hypotheses.py"),
               "--validation-results", str(validation_results_csv),
               "--output-dir", str(summary_dir), "--dataset-name", dataset_name]
        ok, out, err, dur = run_command(cmd)
        results["status"]["step8"] = ok
        results["step8_duration"] = dur
        if not ok:
            results["errors"]["step8"] = err
            results["step8_skipped"] = True  # will use fallback

    progress.progress(100, text="Pipeline complete!")
    return results


# ============================================================
# RUN PIPELINE
# ============================================================

if run_pipeline and input_csv_path and dataset_name:
    with st.spinner("Running pipeline..."):
        pipeline_results = execute_pipeline(input_csv_path, dataset_name)
    st.session_state["pipeline_results"] = pipeline_results
    st.session_state["pipeline_dataset"] = dataset_name
    st.balloons()


# ============================================================
# DISPLAY RESULTS IN TABS
# ============================================================

def show_results(dataset_name):
    """Display pipeline results for a dataset using tabs."""
    dataset_root = OUTPUT_ROOT / dataset_name
    if not dataset_root.exists():
        st.info("Select a dataset and click **Run Pipeline** to start.")
        return

    paths = {
        "data_dir": dataset_root / "data",
        "preprocess_dir": dataset_root / "preprocess",
        "associations_dir": dataset_root / "associations",
        "prompt_dir": dataset_root / "prompt",
        "hypotheses_dir": dataset_root / "hypotheses",
        "validation_dir": dataset_root / "validation",
        "summary_dir": dataset_root / "summary",
    }

    # Status bar
    step_names = ["Split", "Preprocess", "Associations", "Prompt", "Hypotheses", "Validation", "Summary"]
    step_dirs = [paths["data_dir"], paths["preprocess_dir"], paths["associations_dir"],
                 paths["prompt_dir"], paths["hypotheses_dir"], paths["validation_dir"], paths["summary_dir"]]
    cols = st.columns(len(step_names))
    for i, (name, d) in enumerate(zip(step_names, step_dirs)):
        icon = ":material/check_circle:" if d.exists() and any(d.iterdir()) else ":material/radio_button_unchecked:"
        cols[i].markdown(f"{icon} **{name}**")

    st.divider()

    # Main tabs
    tab_data, tab_preprocess, tab_assoc, tab_prompt, tab_hypotheses, tab_validation, tab_summary = st.tabs([
        ":material/call_split: Data Split",
        ":material/auto_fix_high: Preprocess",
        ":material/hub: Associations",
        ":material/package_2: Evidence Pack",
        ":material/psychology: Hypotheses",
        ":material/verified: Validation",
        ":material/summarize: Summary",
    ])

    # --- TAB 1: DATA SPLIT ---
    with tab_data:
        train_csv = paths["data_dir"] / f"{dataset_name}_train.csv"
        test_csv = paths["data_dir"] / f"{dataset_name}_test.csv"
        df_train = load_csv_safe(train_csv)
        df_test = load_csv_safe(test_csv)

        if df_train is not None and df_test is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Train rows", f"{len(df_train):,}")
            c2.metric("Test rows", f"{len(df_test):,}")
            c3.metric("Split ratio", f"{len(df_test)/(len(df_train)+len(df_test))*100:.0f}% test")

            fig = go.Figure(data=[
                go.Bar(name="Train", x=["Dataset"], y=[len(df_train)], marker_color="#636EFA"),
                go.Bar(name="Test", x=["Dataset"], y=[len(df_test)], marker_color="#EF553B"),
            ])
            fig.update_layout(barmode="stack", title="Train/Test Split", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Raw data preview
            st.subheader("Raw Data Preview")
            df_raw = pd.read_csv(input_csv_path) if input_csv_path else df_train
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(df_raw):,}")
            col2.metric("Columns", f"{df_raw.shape[1]}")
            col3.metric("Memory", f"{df_raw.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
            st.dataframe(df_raw.head(50), use_container_width=True, height=300)
        else:
            st.info("No split data available. Run the pipeline first.")

    # --- TAB 2: PREPROCESS ---
    with tab_preprocess:
        profile_path = paths["preprocess_dir"] / f"{dataset_name}_train_profile.json"
        summary_path = paths["preprocess_dir"] / f"{dataset_name}_train_preprocess_summary.json"
        profile = load_json_safe(profile_path)
        preprocess_summary = load_json_safe(summary_path)

        if preprocess_summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input cols", preprocess_summary.get("input_cols"))
            c2.metric("Output cols", preprocess_summary.get("output_cols"))
            c3.metric("Dropped", len(preprocess_summary.get("dropped_columns", [])))
            c4.metric("Engineered", len(preprocess_summary.get("engineered_columns", [])))

            if preprocess_summary.get("dropped_columns"):
                st.info(f"Dropped columns: {', '.join(preprocess_summary['dropped_columns'])}")

        if profile:
            col_a, col_b = st.columns(2)
            with col_a:
                feature_types = [v.get("feature_type", "unknown") for v in profile.values()]
                ft_counts = pd.Series(feature_types).value_counts()
                fig = px.pie(values=ft_counts.values, names=ft_counts.index, title="Feature Types")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Missing rate bar chart
                missing_data = [(col, info.get("missing_rate", 0)) for col, info in profile.items()
                                if info.get("missing_rate", 0) > 0]
                if missing_data:
                    miss_df = pd.DataFrame(missing_data, columns=["Column", "Missing Rate"])
                    miss_df = miss_df.sort_values("Missing Rate", ascending=False).head(20)
                    fig = px.bar(miss_df, x="Column", y="Missing Rate", title="Top Missing Columns")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values detected.")

            # Profile table
            st.subheader("Feature Profile")
            profile_rows = []
            for col, info in profile.items():
                profile_rows.append({
                    "Column": col,
                    "Type": info.get("feature_type", ""),
                    "Missing %": round(info.get("missing_rate", 0) * 100, 2),
                    "Unique": info.get("n_unique", 0),
                    "Skewness": info.get("skewness", None),
                    "Outlier %": round((info.get("outlier_rate") or 0) * 100, 2),
                })
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, height=400)
        else:
            st.info("No preprocessing results available. Run the pipeline first.")

    # --- TAB 3: ASSOCIATIONS ---
    with tab_assoc:
        all_fdr = load_csv_safe(paths["associations_dir"] / f"{dataset_name}_train_all_associations_fdr.csv")
        filtered = load_csv_safe(paths["associations_dir"] / f"{dataset_name}_train_associations_filtered.csv")
        num_num = load_csv_safe(paths["associations_dir"] / f"{dataset_name}_train_num_num_raw.csv")
        num_cat = load_csv_safe(paths["associations_dir"] / f"{dataset_name}_train_num_cat_raw.csv")
        cat_cat = load_csv_safe(paths["associations_dir"] / f"{dataset_name}_train_cat_cat_raw.csv")

        if all_fdr is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Num-Num", len(num_num) if num_num is not None else 0)
            c2.metric("Num-Cat", len(num_cat) if num_cat is not None else 0)
            c3.metric("Cat-Cat", len(cat_cat) if cat_cat is not None else 0)
            c4.metric("Significant", len(filtered) if filtered is not None else 0)

            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Effect Sizes", "Filtered Table", "P-values"])

            with sub_tab1:
                if "effect" in all_fdr.columns and "type" in all_fdr.columns:
                    fig = px.histogram(all_fdr.dropna(subset=["effect"]), x="effect", color="type",
                                       nbins=50, title="Effect Size Distribution")
                    st.plotly_chart(fig, use_container_width=True)

            with sub_tab2:
                if filtered is not None and not filtered.empty:
                    display = ["var1", "var2", "type", "test", "effect", "effect_name", "adjusted_p", "n"]
                    available = [c for c in display if c in filtered.columns]
                    st.dataframe(filtered[available].round(4), use_container_width=True, height=400)
                else:
                    st.info("No significant associations after filtering.")

            with sub_tab3:
                if "p_value" in all_fdr.columns:
                    fig = px.histogram(all_fdr.dropna(subset=["p_value"]), x="p_value", nbins=50,
                                       title="P-value Distribution")
                    fig.add_vline(x=alpha, line_dash="dash", line_color="red", annotation_text=f"α={alpha}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No association results available. Run the pipeline first.")

    # --- TAB 4: EVIDENCE PACK ---
    with tab_prompt:
        evidence_pack = load_json_safe(paths["prompt_dir"] / f"{dataset_name}_train_evidence_pack.json")
        prompt_metadata = load_json_safe(paths["prompt_dir"] / f"{dataset_name}_train_prompt_metadata.json")

        if evidence_pack:
            c1, c2, c3 = st.columns(3)
            c1.metric("Edges selected", len(evidence_pack.get("edges", [])))
            c2.metric("Variables", len(evidence_pack.get("variables", {})))
            c3.metric("Prompt chars", f"{prompt_metadata.get('prompt_char_count', 0):,}" if prompt_metadata else "N/A")

            edges = evidence_pack.get("edges", [])
            if edges:
                edge_rows = []
                for e in edges:
                    edge_rows.append({
                        "Edge": e.get("edge_id"),
                        "Var1": e.get("var1"),
                        "Var2": e.get("var2"),
                        "Type": e.get("type"),
                        "Effect": round(e.get("effect", 0), 4),
                        "Size": e.get("effect_size"),
                        "Direction": e.get("direction"),
                    })
                st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)

            # Variables info
            variables = evidence_pack.get("variables", {})
            if variables:
                st.subheader("Variables in Evidence Pack")
                var_rows = []
                for vname, vinfo in variables.items():
                    var_rows.append({"Variable": vname, **{k: v for k, v in vinfo.items() if k != "issues"}})
                st.dataframe(pd.DataFrame(var_rows), use_container_width=True)
        else:
            st.info("No evidence pack available. Run the pipeline first.")

    # --- TAB 5: HYPOTHESES ---
    with tab_hypotheses:
        hypotheses_path = paths["hypotheses_dir"] / f"{dataset_name}_train_hypotheses.json"
        hypotheses_data = load_json_safe(hypotheses_path)

        if hypotheses_data:
            hypotheses_list = hypotheses_data.get("hypotheses", [])
            st.metric("Generated Hypotheses", len(hypotheses_list))

            for h in hypotheses_list:
                with st.container(border=True):
                    st.markdown(f"**{h.get('id', '')}** — {h.get('relationship_type', '')}")
                    st.write(h.get("claim", ""))

                    # Validation spec: outcome ~ predictors + controls
                    val_spec = h.get("validation_spec", {})
                    outcome = val_spec.get("outcome", "")
                    predictors = val_spec.get("predictors", [])
                    controls = val_spec.get("controls", [])
                    formula = val_spec.get("model_formula", "")

                    if formula:
                        formula_display = formula
                    elif outcome and predictors:
                        parts = " + ".join(predictors)
                        if controls:
                            parts += " + " + " + ".join(controls)
                        formula_display = f"{outcome} ~ {parts}"
                    else:
                        formula_display = ""

                    if formula_display:
                        st.code(formula_display, language=None)

                    c1, c2, c3 = st.columns(3)
                    c1.caption(f"Target: **{outcome}** | Predictors: {', '.join(predictors)}")
                    c2.caption(f"Test: {h.get('test_plan', {}).get('primary_test', '')}")
                    c3.caption(f"Direction: {h.get('direction', '')}")

                    # Supporting evidence (edge IDs)
                    evidence = h.get("supporting_evidence", [])
                    if evidence:
                        st.caption(f"Evidence: {', '.join(evidence)}")
        else:
            st.info("No hypotheses available. Run the pipeline (including LLM step) first.")

    # --- TAB 6: VALIDATION ---
    with tab_validation:
        validation_summary = load_json_safe(paths["validation_dir"] / f"{dataset_name}_train_validation_summary.json")
        validation_results = load_csv_safe(paths["validation_dir"] / f"{dataset_name}_train_validation_results.csv")

        if validation_summary:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Raw", validation_summary.get("generated_hypotheses_raw", 0))
            c2.metric("Testable", validation_summary.get("testable_hypotheses", 0))
            c3.metric("Validated", validation_summary.get("validated_hypotheses_before_bootstrap", 0))
            c4.metric("Final Accepted", validation_summary.get("final_accepted_hypotheses", 0))
            c5.metric("Rejected (pre)", validation_summary.get("rejected_pre_validation", 0))

        if validation_results is not None and not validation_results.empty:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Status", "Effect vs P-value", "Full Table"])

            with sub_tab1:
                if "validation_status" in validation_results.columns:
                    status_counts = validation_results["validation_status"].value_counts()
                    fig = px.bar(x=status_counts.index, y=status_counts.values,
                                 title="Validation Status", labels={"x": "Status", "y": "Count"},
                                 color=status_counts.index)
                    st.plotly_chart(fig, use_container_width=True)

            with sub_tab2:
                if "effect" in validation_results.columns:
                    vr = validation_results.dropna(subset=["effect"]).copy()
                    if not vr.empty:
                        vr["Result"] = vr["final_accepted"].map({True: "Accepted", False: "Rejected"})
                        fig = px.scatter(vr, x="effect", y="p_value", color="Result",
                                         hover_data=["id", "primary_test"],
                                         title="Effect Size vs P-value")
                        fig.add_hline(y=alpha, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

            with sub_tab3:
                display_cols = ["id", "claim", "primary_test", "effect", "p_value",
                                "validation_status", "final_accepted"]
                available = [c for c in display_cols if c in validation_results.columns]
                st.dataframe(validation_results[available].round(4), use_container_width=True, height=400)

            # Final accepted
            final_hypotheses = load_json_safe(paths["validation_dir"] / f"{dataset_name}_train_final_hypotheses.json")
            if final_hypotheses:
                final_list = final_hypotheses.get("hypotheses", [])
                if final_list:
                    st.subheader(":material/emoji_events: Final Accepted Hypotheses")
                    for h in final_list:
                        st.success(f"**{h.get('id', '')}**: {h.get('claim', '')}")
        else:
            st.info("No validation results available. Run the pipeline first.")

    # --- TAB 7: SUMMARY ---
    with tab_summary:
        summary_file = paths["summary_dir"] / f"{dataset_name}_final_summary.md"
        if summary_file.exists():
            st.markdown(summary_file.read_text(encoding="utf-8"))
        else:
            # Generate fallback if validation exists
            if (paths["validation_dir"] / f"{dataset_name}_train_validation_results.csv").exists():
                generate_fallback_summary(paths["validation_dir"], paths["summary_dir"], dataset_name)
            else:
                st.info("No summary available. Run the pipeline first.")


# ============================================================
# MAIN DISPLAY LOGIC
# ============================================================

if input_csv_path and dataset_name:
    dataset_root = OUTPUT_ROOT / dataset_name
    if dataset_root.exists():
        show_results(dataset_name)
    elif not run_pipeline:
        st.info("Select a dataset and click **Run Pipeline** to start the analysis.")
elif not input_csv_path:
    st.info("Select or upload a dataset from the sidebar to begin.")
