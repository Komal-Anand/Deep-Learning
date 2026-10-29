"""
AI-Generated Image Detection Dashboard.
Main Streamlit entry point.
Theme: Deep navy (#0a0f1d, #0f172a) with modern blue (#3b82f6) and violet (#8b5cf6) accents.
"""

import streamlit as st
from pathlib import Path
import config
from models.inference import InferenceEngine
from utils.ui_theme import apply_custom_theme

# Page Configuration
st.set_page_config(
    page_title="AI Image Sentinel | Explainable Deep Learning Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Ultra-Modern Global Theme
apply_custom_theme()

# Initialize Session History
if "history" not in st.session_state:
    st.session_state["history"] = []

# Instantiate Engine for status
engine = InferenceEngine()
model_status = engine.get_model_status()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 8px;">
        <span style="font-size: 1.8rem;">🛡️</span>
        <div>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700;">AI Sentinel</h3>
            <p style="margin:0; font-size:0.75rem; color:#94a3b8;">Forensic Image Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Final-Year B.Tech CSE (DS & ML) Project")
    st.markdown("---")
    st.markdown("#### Neural Checkpoints")
    
    for m_name, m_info in model_status.items():
        if m_info["trained"]:
            st.markdown(f"**{m_name}** <span class='status-badge-trained'>● Active ({m_info['size_mb']} MB)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{m_name}** <span class='status-badge-untrained'>○ Untrained</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔒 **Zero-Retention Guarantee**: Uploaded media resides exclusively in volatile runtime memory.")

# Main Dashboard Hero
st.markdown("""
<div style="margin-top: 10px; margin-bottom: 24px;">
    <span class="hero-badge">⚡ Explainable AI • Deep Learning Forensics • Final Year Project</span>
    <h1 class="hero-gradient-text">AI-Generated Image Detection</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; max-width: 800px; line-height: 1.6; margin-top: 8px;">
        A high-performance research platform distinguishing authentic optical photography from generative 
        media (Diffusion models, GANs) using <strong>EfficientNet-B0 transfer learning</strong> and 
        <strong>Grad-CAM</strong> spatial explainability.
    </p>
</div>
""", unsafe_allow_html=True)

# Status Banner
any_trained = any(m["trained"] for m in model_status.values())
if any_trained:
    st.success("System Ready: Trained deep learning model weights loaded and active for inference.")
else:
    st.warning("Demo / Untrained Mode: Model checkpoints not detected in `artifacts/`. Please run the training pipeline (`python training/train_efficientnet.py`) to generate weights.")

# Architecture & Overview Grid
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="research-card">
        <h3>🎯 Problem Motivation</h3>
        <p style="color: #94a3b8; line-height: 1.6;">
            With the exponential advancement of Latent Diffusion Models (Stable Diffusion, Midjourney, DALL-E 3) 
            and Generative Adversarial Networks (StyleGAN3), hyper-realistic synthetic imagery has become visually 
            indistinguishable to the human eye. 
        </p>
        <p style="color: #94a3b8; line-height: 1.6;">
            This project provides an automated, scientifically reproducible detection framework combining transfer 
            learning via <strong>EfficientNet-B0</strong>, a baseline <strong>Custom CNN</strong>, and 
            <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) to audit spatial visual artifacts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="research-card">
        <h3>🔬 Key Technical Highlights</h3>
        <ul style="color: #94a3b8; line-height: 1.8;">
            <li><strong>Transfer Learning:</strong> Pretrained EfficientNet-B0 backbone fine-tuned for binary classification.</li>
            <li><strong>Explainable AI:</strong> Layer-wise Grad-CAM heatmaps highlight synthetic artifact regions.</li>
            <li><strong>Scientific Rigor:</strong> Zero data leakage via SHA-256 deduplication across splits.</li>
            <li><strong>Robustness Benchmark:</strong> Evaluated under JPEG compression, blur, and downscaling.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Feature Cards
st.markdown("### 🚀 Exploration Modules")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    st.markdown("""
    <div class="research-card">
        <h4>🔍 Live Forensics</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Deep learning detection, Grad-CAM heatmaps, 2D FFT spectral analysis, and ELA.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Detection →", key="btn_detect", use_container_width=True):
        st.switch_page("pages/1_Image_Detection.py")

with f_col2:
    st.markdown("""
    <div class="research-card">
        <h4>📊 Benchmark</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Empirical test set metrics: Confusion matrices, ROC-AUC, and robustness degradation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Evaluation →", key="btn_eval", use_container_width=True):
        st.switch_page("pages/2_Model_Evaluation.py")

with f_col3:
    st.markdown("""
    <div class="research-card">
        <h4>🎮 Turing Challenge</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Interactive viva challenge: Human eye perception pitted against EfficientNet-B0.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Play Challenge →", key="btn_quiz", use_container_width=True):
        st.switch_page("pages/4_Human_vs_AI_Quiz.py")

with f_col4:
    st.markdown("""
    <div class="research-card">
        <h4>📖 Academic Dossier</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Formal methodology, mathematical equations, test cases, and 20 Viva Voce Q&A.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Read Dossier →", key="btn_about", use_container_width=True):
        st.switch_page("pages/3_About_Project.py")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 20px;">
    AI-Generated Image Detection Using Deep Learning and Explainable AI • B.Tech CSE Final-Year Project • Built with PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)

