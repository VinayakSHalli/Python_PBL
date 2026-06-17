import streamlit as st
import numpy as np
from PIL import Image
import math

st.set_page_config(
    page_title="Nyquist–Shannon Image Sampling Explorer",
    layout="wide",
    page_icon="📸"
)

# Improved Custom CSS
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
        color: #64748B;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: #1E3A8A;
    }
    .workflow-step {
        background: #F8FAFC;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: #FFFFFF;
        padding: 1.4rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #E2E8F0;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Understand how sampling density affects image reconstruction and aliasing</p>', unsafe_allow_html=True)

# ================== HOW SAMPLING WORKS ==================
st.markdown('<h3 class="section-header">📋 How Sampling Works</h3>', unsafe_allow_html=True)

col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    st.markdown("""
    <div class="workflow-step">
        <strong>1. Original Image</strong><br>
        High-resolution reference (approximating continuous scene)
    </div>
    """, unsafe_allow_html=True)

with col_w2:
    st.markdown("""
    <div class="workflow-step">
        <strong>2. Sampling</strong><br>
        Downsample → Simulate fewer spatial measurements
    </div>
    """, unsafe_allow_html=True)

with col_w3:
    st.markdown("""
    <div class="workflow-step">
        <strong>3. Reconstruction</strong><br>
        Upsample using interpolation method
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Controls
col1, col2 = st.columns([1, 1])

with col1:
    source = st.radio(
        "📷 Image Source",
        ["Upload image", "Checkerboard", "Vertical stripes"],
        horizontal=True
    )

with col2:
    sampling = st.slider(
        "🎛️ Effective Sampling Density (%)",
        5, 100, 100, 5,
        help="Lower = fewer samples → more aliasing & detail loss"
    )

# Image functions (unchanged)
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
    uploaded = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
else:
    if source == "Checkerboard":
        img = checkerboard()
    elif source == "Vertical stripes":
        img = vertical_stripes()

# Interpolation
interp_options = {
    "Nearest": "🟦 Nearest (Blocky)",
    "Bilinear": "🟩 Bilinear (Smooth)",
    "Bicubic": "🟨 Bicubic (Good detail)",
    "Lanczos": "⭐ Lanczos (Sharpest)",
    "Sinc (Ideal Approximation)": "📐 Sinc (Theoretical best)"
}

interp_name = st.selectbox(
    "🔬 Reconstruction Method",
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

    # ================== VISUAL COMPARISON ==================
    st.markdown('<h3 class="section-header">📊 Visual Comparison</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img, caption="**Original**", use_container_width=True)
    with c2:
        st.image(sampled, caption=f"**Sampled** ({sw}×{sh})", use_container_width=True)
    with c3:
        st.image(recon, caption="**Reconstructed**", use_container_width=True)

    st.divider()

    # ================== QUALITY METRICS ==================
    st.markdown('<h3 class="section-header">📈 Quality Metrics</h3>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Sampled Resolution", f"{sw} × {sh}")
    with m2:
        st.metric("Mean Squared Error", f"{m:.2f}")
    with m3:
        st.metric("PSNR (dB)", "∞" if np.isinf(p) else f"{p:.2f}")

    # Insights
    st.divider()
    if source != "Upload image":
        if sampling >= 60:
            st.success("✅ Nyquist condition likely satisfied")
        elif sampling >= 35:
            st.warning("⚠️ Borderline sampling — aliasing may appear")
        else:
            st.error("❌ Severe undersampling — strong aliasing expected")
    else:
        st.info("For real images, lower sampling always loses fine details.")

else:
    st.info("👆 Select a pattern or upload an image to start")

st.caption("Nyquist–Shannon Image Sampling Explorer")
