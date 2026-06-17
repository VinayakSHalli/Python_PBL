import streamlit as st
import numpy as np
from PIL import Image
import math
from scipy.io import wavfile
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="Nyquist–Shannon Sampling Explorer",
    layout="wide",
    page_icon="🔬"
)

# ====================== COMBINED CSS ======================
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subheader {
        text-align: center;
        color: #475569;
        font-size: 1.4rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2.0rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 2rem 0 1rem 0;
    }
    .image-label {
        font-size: 1.85rem !important;
        font-weight: 600;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.6);
    }
    .stSelectbox label, .stRadio label, .stSlider label, .stNumberInput label {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    .stMetric label {
        font-size: 1.45rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2.1rem !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Explore Sampling Theory in Images & Audio Signals</p>', unsafe_allow_html=True)

# ====================== MODE SELECTOR ======================
mode = st.radio(
    "Select Domain to Explore",
    ["🖼️ Image Sampling", "🎵 Audio Sampling"],
    horizontal=True,
    label_visibility="visible"
)

st.divider()

# ====================== IMAGE SAMPLING ======================
if mode == "🖼️ Image Sampling":
    
    st.markdown('<h3 class="section-header">How Sampling Works</h3>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown('<p><strong>1. Original Image</strong><br>High-resolution reference</p>', unsafe_allow_html=True)
    with w2:
        st.markdown('<p><strong>2. Sampling</strong><br>Downsample (fewer spatial samples)</p>', unsafe_allow_html=True)
    with w3:
        st.markdown('<p><strong>3. Reconstruction</strong><br>Upscale with interpolation</p>', unsafe_allow_html=True)

    st.divider()

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        source = st.radio(
            "Image Source",
            ["Upload image", "Checkerboard", "Vertical stripes"],
            horizontal=True
        )
    with col_ctrl2:
        sampling = st.slider(
            "Effective Sampling Density (%)",
            min_value=5, max_value=100, value=100, step=5
        )

    # Image functions
    def checkerboard(size=512, block=8):
        y, x = np.indices((size, size))
        arr = (((x // block) + (y // block)) % 2) * 255
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    def vertical_stripes(size=512, stripe=4):
        arr = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            arr[:, i] = 255 if (i // stripe) % 2 == 0 else 0
        return Image.fromarray(arr).convert("RGB")

    img = None
    if source == "Upload image":
        uploaded = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        if source == "Checkerboard":
            img = checkerboard()
        elif source == "Vertical stripes":
            img = vertical_stripes()

    interp_options = {
        "Nearest": "Nearest (Blocky)",
        "Bilinear": "Bilinear (Smooth)",
        "Bicubic": "Bicubic (Good detail)",
        "Lanczos": "Lanczos (Sharpest)",
        "Sinc (Ideal Approximation)": "Sinc (Theoretical best)"
    }

    interp_name = st.selectbox(
        "Reconstruction Method",
        list(interp_options.keys()),
        format_func=lambda x: interp_options[x]
    )

    if img is not None:
        w, h = img.size
        sw = max(1, int(w * sampling / 100))
        sh = max(1, int(h * sampling / 100))

        sampled = img.resize((sw, sh), Image.Resampling.BOX)

        if interp_name == "Sinc (Ideal Approximation)":
            recon = sampled.resize((w, h), Image.Resampling.LANCZOS)
        else:
            interp_map = {
                "Nearest": Image.Resampling.NEAREST,
                "Bilinear": Image.Resampling.BILINEAR,
                "Bicubic": Image.Resampling.BICUBIC,
                "Lanczos": Image.Resampling.LANCZOS
            }
            recon = sampled.resize((w, h), interp_map[interp_name])

        m = np.mean((np.array(img, dtype=np.float32) - np.array(recon, dtype=np.float
