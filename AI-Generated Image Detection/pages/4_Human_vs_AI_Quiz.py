"""
Streamlit Page 4: "Human vs. AI" Turing Test Forensic Challenge.
An interactive viva evaluation game:
- Evaluator attempts to identify Real vs AI-Generated images.
- Compares Human visual perception accuracy directly against the EfficientNet-B0 model.
"""

import streamlit as st
import random
from pathlib import Path
from PIL import Image

import config
from models.inference import InferenceEngine
from utils.ui_theme import apply_custom_theme

st.set_page_config(
    page_title="Human vs AI Quiz | AI Image Sentinel",
    page_icon="🎮",
    layout="wide"
)

# Apply Global Theme
apply_custom_theme()

# State initialization
if "quiz_round" not in st.session_state:
    st.session_state["quiz_round"] = 1
if "human_score" not in st.session_state:
    st.session_state["human_score"] = 0
if "ai_score" not in st.session_state:
    st.session_state["ai_score"] = 0
if "current_image_path" not in st.session_state:
    st.session_state["current_image_path"] = None
if "current_truth" not in st.session_state:
    st.session_state["current_truth"] = None
if "revealed" not in st.session_state:
    st.session_state["revealed"] = False
if "ai_prediction" not in st.session_state:
    st.session_state["ai_prediction"] = None
if "user_guess" not in st.session_state:
    st.session_state["user_guess"] = None

@st.cache_resource
def get_engine():
    return InferenceEngine()

engine = get_engine()

# Gather test images
test_dir = config.TEST_DIR
real_images = list((test_dir / "Real").glob("*.*")) if (test_dir / "Real").exists() else []
ai_images = list((test_dir / "AI-Generated").glob("*.*")) if (test_dir / "AI-Generated").exists() else []
all_test_images = [(p, "Real") for p in real_images] + [(p, "AI-Generated") for p in ai_images]

st.title("🎮 Human vs. AI: The Visual Turing Challenge")
st.caption("Can the human eye outperform deep convolutional neural networks? Test your perceptual limits against EfficientNet-B0.")

# Scoreboard
c_score1, c_score2, c_score3 = st.columns(3)
with c_score1:
    st.markdown(f"""
    <div class="score-box">
        <div style="color:#94a3b8; font-size:0.9rem;">ROUND</div>
        <div class="score-number">{st.session_state['quiz_round']}</div>
    </div>
    """, unsafe_allow_html=True)

with c_score2:
    st.markdown(f"""
    <div class="score-box">
        <div style="color:#10b981; font-size:0.9rem;">YOUR HUMAN SCORE</div>
        <div class="score-number" style="color:#34d399;">{st.session_state['human_score']}</div>
    </div>
    """, unsafe_allow_html=True)

with c_score3:
    st.markdown(f"""
    <div class="score-box">
        <div style="color:#a78bfa; font-size:0.9rem;">EFFICIENTNET-B0 SCORE</div>
        <div class="score-number" style="color:#c084fc;">{st.session_state['ai_score']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if not all_test_images:
    st.warning("No test images available. Please run `python training/prepare_data.py` first.")
    st.stop()

# Pick next image if needed
if st.session_state["current_image_path"] is None:
    chosen_path, truth = random.choice(all_test_images)
    st.session_state["current_image_path"] = chosen_path
    st.session_state["current_truth"] = truth
    st.session_state["revealed"] = False
    st.session_state["ai_prediction"] = None
    st.session_state["user_guess"] = None

current_img = Image.open(st.session_state["current_image_path"]).convert("RGB")
truth = st.session_state["current_truth"]

# Presentation
col_img, col_interaction = st.columns([1, 1])

with col_img:
    st.image(current_img, caption="Mystery Image (Look closely at textures and edges)", use_container_width=True)

with col_interaction:
    st.markdown("### 🤔 Make Your Prediction")
    st.write("Inspect the lighting, textures, shadows, and subtle high-frequency details. Is this an authentic optical photo or AI-generated media?")

    if not st.session_state["revealed"]:
        btn_real = st.button("📷 I think it's a REAL PHOTOGRAPH", use_container_width=True)
        btn_ai = st.button("🤖 I think it's AI-GENERATED", use_container_width=True)

        user_choice = None
        if btn_real:
            user_choice = "Real"
        elif btn_ai:
            user_choice = "AI-Generated"

        if user_choice is not None:
            st.session_state["user_guess"] = user_choice
            st.session_state["revealed"] = True

            # Model prediction
            res = engine.analyze_image(current_img, model_name="EfficientNet-B0", generate_cam=False)
            ai_pred = res["predicted_class"] if res["success"] else "Unknown"
            st.session_state["ai_prediction"] = res

            # Update scores
            if user_choice == truth:
                st.session_state["human_score"] += 1
            if ai_pred == truth:
                st.session_state["ai_score"] += 1

            st.rerun()

    else:
        # Results Reveal
        user_guess = st.session_state["user_guess"]
        res = st.session_state["ai_prediction"]
        ai_pred = res["predicted_class"] if res["success"] else "Error"
        conf = res["confidence"] if res["success"] else 0.0

        st.markdown(f"#### Ground Truth: **{truth}**")

        # Human result
        if user_guess == truth:
            st.success(f"✅ **You Were Correct!** You guessed {user_guess}.")
        else:
            st.error(f"❌ **Human Eye Deceived!** You guessed {user_guess}, but the ground truth was {truth}.")

        # AI Detector result
        if ai_pred == truth:
            st.info(f"🤖 **EfficientNet-B0 was Correct!** Predicted {ai_pred} with {conf*100:.1f}% confidence.")
        else:
            st.warning(f"🤖 EfficientNet-B0 guessed {ai_pred} ({conf*100:.1f}%).")

        st.markdown("---")

        col_next, col_restart = st.columns(2)
        with col_next:
            if st.button("Next Challenge →", use_container_width=True):
                st.session_state["quiz_round"] += 1
                st.session_state["current_image_path"] = None
                st.rerun()

        with col_restart:
            if st.button("Reset Challenge", use_container_width=True):
                st.session_state["quiz_round"] = 1
                st.session_state["human_score"] = 0
                st.session_state["ai_score"] = 0
                st.session_state["current_image_path"] = None
                st.rerun()

