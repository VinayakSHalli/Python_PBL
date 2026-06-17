import streamlit as st
import numpy as np
from PIL import Image
import math

st.set_page_config(
    page_title="Nyquist–Shannon Image Sampling Explorer",
    layout="wide",
    page_icon="📸"
)

# Minimal CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
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
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 1.8rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Visualize sampling, reconstruction, and aliasing in images</p>', unsafe_allow_html=True)

# ================== HOW SAMPLING WORKS ==================
st.markdown('<h3 class="section-header">📋 How Sampling Works</h3>', unsafe_allow_html=True)

w1, w2, w3 = st.columns(3)

with w1:
    st.markdown("**1. Original Image**")
    st.markdown("High-resolution reference (stands in for a continuous scene)")

with w2:
    st.markdown("**2. Sampling**")
    st.markdown("Downsample the image to simulate **fewer spatial samples**")

with w3:
    st.markdown("**3. Reconstruction**")
    st.markdown("Upscale back using different interpolation methods")

st.divider()

# Controls
col_ctrl1, col_ctrl2 = st.columns([1, 1])

with col_ctrl1:
    source = st.radio(
        "📷 Image Source",
        ["Upload image", "Checkerboard", "Vertical stripes"],
        horizontal=True
    )

with col_ctrl2:
    sampling = st.slider(
        "🎛️ Effective Sampling Density (%)",
        min_value=5,
        max_value=100,
        value=100,
        step=5,
        help="Lower values = fewer samples → more aliasing and detail loss"
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

# Reconstruction method
interp_options = {
    "Nearest": "🟦 Nearest (Blocky)",
    "Bilinear": "🟩 Bilinear (Smooth)",
    "Bicubic": "🟨 Bicubic (Good
