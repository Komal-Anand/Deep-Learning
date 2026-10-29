"""
UI Theme Module: Ultra-Modern Cyber-Research Aesthetic.
Provides a unified design system with:
- Ambient mesh gradients & obsidian dark background
- Glassmorphic frosted cards with micro-glows
- Polished typography (Inter & Space Grotesk)
- Neon status indicators, custom file dropzones, and styled progress bars.
"""

import streamlit as st


def get_custom_css() -> str:
    return """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">

<style>
    /* ===================================================
       1. GLOBAL ROOT VARIABLES & BACKGROUND MESH
       =================================================== */
    :root {
        --bg-main: #07090e;
        --bg-card: rgba(15, 23, 42, 0.65);
        --bg-card-hover: rgba(20, 30, 55, 0.75);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.4);
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-violet: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(139, 92, 246, 0.06) 0px, transparent 60%) !important;
        background-attachment: fixed !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Headings font */
    h1, h2, h3 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        letter-spacing: -0.03em !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* ===================================================
       2. GLASSMORPHIC RESEARCH CARDS
       =================================================== */
    .research-card {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    .research-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.8), 0 0 25px rgba(99, 102, 241, 0.15) !important;
    }

    /* Gradient Hero Text */
    .hero-gradient-text {
        background: linear-gradient(135deg, #ffffff 10%, #93c5fd 50%, #c084fc 90%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        letter-spacing: -0.04em !important;
        line-height: 1.15 !important;
    }

    /* Hero Subtitle Badge */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 12px;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.2);
    }

    /* ===================================================
       3. PREDICTION VERDICT BOXES (NEON GLOW)
       =================================================== */
    .prediction-box-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%) !important;
        border: 2px solid #10b981 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        text-align: center !important;
        color: #ecfdf5 !important;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25), inset 0 0 15px rgba(16, 185, 129, 0.1) !important;
        animation: glowReal 3s infinite alternate !important;
    }

    .prediction-box-ai {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        border: 2px solid #8b5cf6 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        text-align: center !important;
        color: #f5f3ff !important;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.25), inset 0 0 15px rgba(139, 92, 246, 0.1) !important;
        animation: glowAI 3s infinite alternate !important;
    }

    @keyframes glowReal {
        from { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
        to { box-shadow: 0 0 35px rgba(16, 185, 129, 0.4); }
    }
    @keyframes glowAI {
        from { box-shadow: 0 0 20px rgba(139, 92, 246, 0.2); }
        to { box-shadow: 0 0 35px rgba(139, 92, 246, 0.4); }
    }

    /* Status Badges */
    .status-badge-trained {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2) !important;
    }

    .status-badge-untrained {
        background: rgba(239, 68, 68, 0.15) !important;
        color: #f87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
    }

    /* Metric Card */
    .metric-card {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 14px !important;
        padding: 18px 14px !important;
        text-align: center !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    .metric-val {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #60a5fa !important;
        margin-top: 4px !important;
        letter-spacing: -0.02em !important;
    }
    .metric-lbl {
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
    }

    /* ===================================================
       4. BUTTONS & INTERACTIVE ELEMENTS
       =================================================== */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.25s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.55), 0 0 12px rgba(139, 92, 246, 0.35) !important;
        transform: translateY(-1px) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* File Uploader Custom Styling */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(6, 182, 212, 0.6) !important;
        background: rgba(15, 23, 42, 0.7) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 6px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.2) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%) !important;
        border-radius: 10px !important;
    }

    /* Sidebar Glass Finish */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 29, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(20px) !important;
    }
</style>
"""


def apply_custom_theme():
    """Injects the global ultra-modern cyber-research design system."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

