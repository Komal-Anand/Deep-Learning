"""
Streamlit Page 2: Research Benchmark & Model Evaluation.
Loads empirical results from reports/evaluation_metrics.json and visualizes figures.
"""

import streamlit as st
import json
from pathlib import Path
import config
from utils.ui_theme import apply_custom_theme

st.set_page_config(
    page_title="Model Evaluation | AI Image Sentinel",
    page_icon="📊",
    layout="wide"
)

# Apply Global Theme
apply_custom_theme()

st.markdown("""
<div style="margin-top: 5px; margin-bottom: 20px;">
    <span class="hero-badge">📊 Empirical Benchmarks</span>
    <h1 style="margin: 0; font-size: 2.2rem;">Model Benchmark & Empirical Evaluation</h1>
    <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px;">
        Quantitative performance metrics and robustness perturbations evaluated strictly on quarantined test data.
    </p>
</div>
""", unsafe_allow_html=True)

metrics_file = config.EVALUATION_METRICS_PATH

if not metrics_file.exists():
    st.warning("⚠️ No evaluation metrics found on disk.")
    st.info(
        "To evaluate the models and compute empirical test metrics, run the evaluation script:\n\n"
        "```bash\n"
        "python training/evaluate.py\n"
        "```\n"
        "This will test both architectures on held-out test data, compute accuracy, confusion matrices, "
        "ROC-AUC curves, and robustness perturbations, saving all plots to `reports/`."
    )
else:
    with open(metrics_file, "r") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    test_size = data.get("test_dataset_size", 0)

    st.markdown(f"**Test Set Distribution**: {test_size} balanced held-out images with zero training overlap.")

    # Model Tabs
    model_names = list(models_data.keys())
    if not model_names:
        st.info("Metrics file is empty. Please run `python training/evaluate.py`.")
    else:
        # Comparison Overview
        if len(model_names) > 1:
            st.markdown("### ⚔️ Comparative Architecture Summary")
            comp_chart_path = config.REPORTS_DIR / "model_comparison_chart.png"
            if comp_chart_path.exists():
                st.image(str(comp_chart_path), caption="Empirical Metric Comparison (Held-Out Test Set)", use_container_width=True)

        tabs = st.tabs([f"📌 {m}" for m in model_names])

        for tab, m_name in zip(tabs, model_names):
            with tab:
                m_info = models_data[m_name]

                # Metric Cards Row
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Accuracy</div>
                        <div class="metric-val">{m_info.get('accuracy', 0)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Precision</div>
                        <div class="metric-val">{m_info.get('precision', 0)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Recall</div>
                        <div class="metric-val">{m_info.get('recall', 0)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">F1-Score</div>
                        <div class="metric-val">{m_info.get('f1_score', 0):.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">ROC-AUC</div>
                        <div class="metric-val">{m_info.get('roc_auc', 0):.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Visualizations Grid
                col_cm, col_roc = st.columns(2)
                prefix = m_name.lower().replace(" ", "_")

                with col_cm:
                    st.markdown("#### Confusion Matrix")
                    cm_img = config.REPORTS_DIR / f"{prefix}_confusion_matrix.png"
                    if cm_img.exists():
                        st.image(str(cm_img), use_container_width=True)
                    else:
                        st.caption("Confusion matrix plot not available.")

                with col_roc:
                    st.markdown("#### ROC Curve")
                    roc_img = config.REPORTS_DIR / f"{prefix}_roc_curve.png"
                    if roc_img.exists():
                        st.image(str(roc_img), use_container_width=True)
                    else:
                        st.caption("ROC curve plot not available.")

                # Training Curves & Robustness Row
                col_train, col_rob = st.columns(2)

                with col_train:
                    st.markdown("#### Training & Validation Dynamics")
                    curves_img = config.REPORTS_DIR / f"{prefix}_training_curves.png"
                    if curves_img.exists():
                        st.image(str(curves_img), use_container_width=True)
                    else:
                        st.caption("Training history curves not available.")

                with col_rob:
                    st.markdown("#### Robustness Perturbation Evaluation")
                    rob_img = config.REPORTS_DIR / f"{prefix}_robustness_chart.png"
                    if rob_img.exists():
                        st.image(str(rob_img), use_container_width=True)
                    else:
                        st.caption("Robustness chart not available.")

                # Detailed Classification Report
                st.markdown("#### Detailed Classification Report")
                st.code(m_info.get("classification_report", "N/A"), language="text")

