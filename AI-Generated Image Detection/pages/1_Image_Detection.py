"""
Streamlit Page 1: Live Image Detection & Multi-Modal Explainability.
Features:
- Genuine deep learning inference with Softmax confidence distribution.
- Interactive Grad-CAM spatial activation maps.
- 2D Fast Fourier Transform (FFT) Log Magnitude Spectrum analysis.
- Error Level Analysis (ELA) for compression inconsistency forensics.
- Forensic EXIF & AI Generation Prompt Inspector.
- One-click downloadable Forensic Audit Certificate.
"""

import streamlit as st
from PIL import Image
import datetime
import config
from models.inference import InferenceEngine
from utils.ui_theme import apply_custom_theme
from utils.forensics import (
    compute_fft_spectrum,
    compute_error_level_analysis,
    extract_image_metadata,
    generate_forensic_certificate
)

st.set_page_config(
    page_title="Image Detection & Forensics | AI Image Sentinel",
    page_icon="🔍",
    layout="wide"
)

# Apply Global Theme
apply_custom_theme()

# Session state history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Engine instantiation
@st.cache_resource
def get_inference_engine():
    return InferenceEngine()

engine = get_inference_engine()

st.markdown("""
<div style="margin-top: 5px; margin-bottom: 20px;">
    <span class="hero-badge">🔬 Forensic Intelligence Lab</span>
    <h1 style="margin: 0; font-size: 2.2rem;">Live AI Image Detection & Multi-Modal Forensics</h1>
    <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px;">
        Comprehensive spatial gradient, 2D spectral, and compression error analysis of digital imagery.
    </p>
</div>
""", unsafe_allow_html=True)

# Controls & Configuration
col_config, col_status = st.columns([2, 1])

with col_config:
    model_choice = st.selectbox(
        "Select Detection Model Architecture:",
        options=["EfficientNet-B0 (Recommended - Transfer Learning)", "Custom CNN (Baseline Architecture)"]
    )
    normalized_model_name = "EfficientNet-B0" if "EfficientNet" in model_choice else "Custom CNN"

with col_status:
    is_trained = engine.is_model_trained(normalized_model_name)
    if is_trained:
        st.markdown(f"**Model Status**: <span style='color:#34d399;'>● Weights Active</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Model Status**: <span style='color:#f87171;'>● Untrained (Missing .pth)</span>", unsafe_allow_html=True)

# Upload Area
uploaded_file = st.file_uploader(
    "Choose an image to analyze (JPG, JPEG, PNG, WEBP — Max 15MB)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Image is processed locally in memory. No data is stored externally."
)

if uploaded_file is not None:
    # Size check
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > config.MAX_IMAGE_SIZE_BYTES:
        st.error(f"File size exceeds maximum permitted limit ({config.MAX_IMAGE_SIZE_BYTES / (1024*1024):.1f} MB).")
        st.stop()

    try:
        preview_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Cannot decode image file. File may be corrupted. Error: {str(e)}")
        st.stop()

    col_preview, col_action = st.columns([1, 1])

    with col_preview:
        st.image(preview_img, caption=f"Uploaded: {uploaded_file.name} ({preview_img.width}x{preview_img.height})", use_container_width=True)

    with col_action:
        st.markdown("### Forensic Inspection Options")
        run_cam = st.checkbox("Generate Grad-CAM Spatial Heatmap", value=True, help="Calculates gradient activations on the final convolutional layer.")
        run_fft = st.checkbox("Compute 2D FFT Frequency Magnitude Spectrum", value=True, help="Detects high-frequency deconvolution checkerboard artifacts.")
        run_ela = st.checkbox("Perform Error Level Analysis (ELA)", value=True, help="Highlights compression discrepancies across image regions.")
        
        btn_analyze = st.button("🚀 Run Full Forensic Analysis", use_container_width=True)
        btn_reset = st.button("🔄 Reset Image", use_container_width=True)

        if btn_reset:
            st.rerun()

    if btn_analyze:
        if not is_trained:
            st.error(
                f"Cannot run genuine prediction: Model weights for **{normalized_model_name}** are not found on disk. "
                f"Please train the model using `python training/train_efficientnet.py` (or `train_cnn.py`)."
            )
        else:
            with st.spinner(f"Analyzing multi-modal representations using {normalized_model_name}..."):
                result = engine.analyze_image(
                    image_input=preview_img,
                    model_name=normalized_model_name,
                    generate_cam=run_cam
                )

            if not result["success"]:
                st.error(f"Detection error: {result['error']}")
            else:
                pred_class = result["predicted_class"]
                conf = result["confidence"]
                probs = result["probabilities"]
                explanation = result["explanation"]

                # Log to session history
                st.session_state["history"].append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "filename": uploaded_file.name,
                    "model": normalized_model_name,
                    "prediction": pred_class,
                    "confidence": f"{conf * 100:.1f}%",
                    "thumbnail": preview_img.resize((64, 64))
                })

                st.markdown("---")
                st.markdown("## 🎯 Classification Verdict")

                # Verdict Box
                v_col1, v_col2 = st.columns([1, 1])

                with v_col1:
                    if pred_class == "Real":
                        st.markdown(f"""
                        <div class="prediction-box-real">
                            <h2 style="margin:0; color:#34d399;">📷 Real Photograph</h2>
                            <h3 style="margin:5px 0; color:#a7f3d0;">Confidence: {conf*100:.1f}%</h3>
                            <p style="margin:0; font-size:0.9rem; color:#ecfdf5;">Matches natural optical camera sensor representations</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box-ai">
                            <h2 style="margin:0; color:#c084fc;">🤖 AI-Generated Image</h2>
                            <h3 style="margin:5px 0; color:#e9d5ff;">Confidence: {conf*100:.1f}%</h3>
                            <p style="margin:0; font-size:0.9rem; color:#f5f3ff;">Exhibits characteristic generative synthetic artifacts</p>
                        </div>
                        """, unsafe_allow_html=True)

                with v_col2:
                    st.markdown("#### Softmax Class Probabilities")
                    for c_name, c_prob in probs.items():
                        st.write(f"**{c_name}**: {c_prob*100:.1f}%")
                        st.progress(c_prob)

                    # Download Certificate Button
                    cert_text = generate_forensic_certificate(result, uploaded_file.name, file_bytes)
                    st.download_button(
                        label="📄 Download Forensic Audit Certificate",
                        data=cert_text,
                        file_name=f"Forensic_Report_{uploaded_file.name}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

                st.markdown("---")
                st.markdown("## 🔬 Multi-Modal Forensic Suite")

                tab_cam, tab_fft, tab_ela, tab_meta = st.tabs([
                    "🔥 Grad-CAM Spatial Heatmap",
                    "🌈 2D FFT Frequency Spectrum",
                    "🎚️ Error Level Analysis (ELA)",
                    "🔍 Forensic Metadata & Prompts"
                ])

                with tab_cam:
                    if run_cam and result.get("gradcam_overlay") is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(result["original_image"], caption="1. Original Input Image", use_container_width=True)
                        with c2:
                            st.image(result["gradcam_overlay"], caption="2. Grad-CAM Activation Overlay", use_container_width=True)

                        st.markdown("""
                        <div class="research-card">
                            <h4>🧠 Interpretability Insight</h4>
                            <p style="color: #cbd5e1; line-height: 1.6;">
                        """ + explanation["technical_summary"] + """
                            </p>
                            <blockquote style="border-left: 3px solid #8b5cf6; padding-left: 12px; margin-top: 10px; color: #94a3b8; font-size: 0.9rem;">
                        """ + explanation["academic_note"] + """
                            </blockquote>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Grad-CAM generation was disabled for this run.")

                with tab_fft:
                    if run_fft:
                        fft_img, magnitude = compute_fft_spectrum(preview_img)
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            st.image(fft_img, caption="2D Fast Fourier Transform (Log Magnitude Spectrum)", use_container_width=True)
                        with f_col2:
                            st.markdown("""
                            <div class="research-card">
                                <h4>📡 Frequency Domain Forensic Analysis</h4>
                                <p style="color: #cbd5e1; line-height: 1.6;">
                                    Natural optical photography follows a power-law frequency distribution where high frequencies 
                                    decay smoothly away from the central low-frequency DC component.
                                </p>
                                <p style="color: #cbd5e1; line-height: 1.6;">
                                    <strong>Generative Artifacts:</strong> Latent diffusion decoders and GAN upsampling layers 
                                    typically leave distinct <em>azimuthal spikes</em>, concentric periodic rings, or grid-like 
                                    energy concentrations in high-frequency spectral corners.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("FFT spectral analysis was disabled.")

                with tab_ela:
                    if run_ela:
                        ela_img = compute_error_level_analysis(preview_img)
                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            st.image(ela_img, caption="Error Level Analysis (ELA at 90% Q)", use_container_width=True)
                        with e_col2:
                            st.markdown("""
                            <div class="research-card">
                                <h4>🎚️ Compression Disparity Forensics</h4>
                                <p style="color: #cbd5e1; line-height: 1.6;">
                                    Error Level Analysis saves the image at a known 90% JPEG quality level and measures pixel 
                                    delta values against the original uncompressed bitmap.
                                </p>
                                <p style="color: #cbd5e1; line-height: 1.6;">
                                    <strong>Interpretation:</strong> Camera sensors exhibit uniform compression noise across 
                                    the entire frame. Synthetically generated or edited regions frequently reveal stark local 
                                    compression disparities, glowing brightly under ELA.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("ELA was disabled.")

                with tab_meta:
                    metadata = extract_image_metadata(preview_img)
                    st.markdown("#### Embedded Image Metadata & Provenance")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"<span class='meta-badge'>Format: <strong>{metadata['format']}</strong></span>", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"<span class='meta-badge'>Dimensions: <strong>{metadata['dimensions']}</strong></span>", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"<span class='meta-badge'>Color Mode: <strong>{metadata['mode']}</strong></span>", unsafe_allow_html=True)

                    if metadata["is_ai_metadata_detected"]:
                        st.warning("⚠️ Synthetic Generation Metadata Detected in Image Chunks!")
                        for k, v in metadata["ai_prompt_info"].items():
                            st.markdown(f"**Chunk `{k}`:**")
                            st.code(v, language="text")
                    else:
                        st.info("No explicit generative prompt chunks found in image header. (Most social media platforms strip metadata upon upload, requiring deep learning detection).")

                    if metadata["camera_exif"]:
                        st.success("📷 Authentic Camera EXIF Data Found:")
                        st.json(metadata["camera_exif"])

# Session Analysis History
if len(st.session_state["history"]) > 0:
    st.markdown("---")
    st.markdown("### 📜 Session Analysis History")
    st.caption("Recent predictions during this browser session. (Images remain in volatile memory only).")

    for item in reversed(st.session_state["history"][-5:]):
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([1, 2, 2, 2, 1])
        with h_col1:
            st.image(item["thumbnail"], width=50)
        with h_col2:
            st.write(f"**{item['filename']}**")
        with h_col3:
            st.write(f"Model: {item['model']}")
        with h_col4:
            st.write(f"**{item['prediction']}** ({item['confidence']})")
        with h_col5:
            st.caption(item["timestamp"])
