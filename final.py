import streamlit as st
import numpy as np
from PIL import Image
import math

st.set_page_config(
    page_title="Nyquist–Shannon Image Sampling Explorer",
    layout="wide",
    page_icon="📸"
)

# Enhanced CSS with larger fonts
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .subheader {
        text-align: center;
        color: #475569;
        font-size: 1.35rem;
        margin-bottom: 2.2rem;
    }
    .section-header {
        font-size: 1.85rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 2rem 0 1.2rem 0;
    }
    .image-label {
        font-size: 1.55rem;
        font-weight: 600;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    /* Overall larger text */
    .stMarkdown, .stMetric, .stRadio, .stSlider, .stSelectbox {
        font-size: 1.1rem;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Visualize sampling, reconstruction, and aliasing in images</p>', unsafe_allow_html=True)

# How Sampling Works
st.markdown('<h3 class="section-header">How Sampling Works</h3>', unsafe_allow_html=True)

w1, w2, w3 = st.columns(3)

with w1:
    st.markdown("**1. Original Image**")
    st.markdown("High-resolution reference (stands in for a continuous scene)")

with w2:
    st.markdown("**2. Sampling**")
    st.markdown("Downsample the image to simulate fewer spatial samples")

with w3:
    st.markdown("**3. Reconstruction**")
    st.markdown("Upscale back using different interpolation methods")

st.divider()

# Controls
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

# Processing
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

    m = np.mean((np.array(img, dtype=np.float32) - np.array(recon, dtype=np.float32)) ** 2)
    p = float("inf") if m == 0 else 20 * math.log10(255.0 / math.sqrt(m))

    # Visual Comparison - Labels above images
    st.markdown('<h3 class="section-header">Visual Comparison</h3>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown('<p class="image-label">Original</p>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    
    with c2:
        st.markdown(f'<p class="image-label">Sampled ({sw} × {sh})</p>', unsafe_allow_html=True)
        st.image(sampled, use_container_width=True)
    
    with c3:
        st.markdown('<p class="image-label">Reconstructed</p>', unsafe_allow_html=True)
        st.image(recon, use_container_width=True)

    st.divider()

    # Quality Metrics
    st.markdown('<h3 class="section-header">Quality Metrics</h3>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Sampled Resolution", f"{sw} × {sh}")
    with m2:
        st.metric("Mean Squared Error (MSE)", f"{m:.2f}")
    with m3:
        st.metric("PSNR (dB)", "∞" if np.isinf(p) else f"{p:.2f}")

    # Insights
    st.divider()
    if source != "Upload image":
        if sampling >= 60:
            st.success("Nyquist condition likely satisfied — Good reconstruction")
        elif sampling >= 35:
            st.warning("Borderline sampling — Some aliasing may appear")
        else:
            st.error("Severe undersampling — Strong aliasing expected")
    else:
        st.info("Note: Real photos lose fine detail at lower sampling rates.")

else:
    st.info("Choose a pattern or upload an image to begin")

st.caption("Nyquist–Shannon Image Sampling & Reconstruction Simulator")
