import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import math

st.set_page_config(
    page_title="Nyquist–Shannon Image Sampling Explorer",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .workflow-step {
        background: #F1F5F9;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    .highlight {
        background: #EFF6FF;
        padding: 1.5rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Visualize how sampling density affects image reconstruction and aliasing</p>', unsafe_allow_html=True)

# Workflow
st.markdown("### 📋 How Sampling Works")
col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.markdown('<div class="workflow-step"><strong>1. Original</strong><br>High-resolution reference image</div>', unsafe_allow_html=True)
with col_w2:
    st.markdown('<div class="workflow-step"><strong>2. Sampling</strong><br>Reduce spatial measurements (downsample)</div>', unsafe_allow_html=True)
with col_w3:
    st.markdown('<div class="workflow-step"><strong>3. Reconstruction</strong><br>Upsample using different interpolators</div>', unsafe_allow_html=True)

st.divider()

# Controls
control_col1, control_col2 = st.columns([1, 1])

with control_col1:
    source = st.radio(
        "📷 **Image Source**",
        ["Upload image", "Checkerboard", "Vertical stripes"],
        horizontal=True,
        label_visibility="visible"
    )

with control_col2:
    sampling = st.slider(
        "🎛️ **Effective Sampling Density (%)**",
        min_value=5,
        max_value=100,
        value=100,
        step=5,
        help="Lower values simulate fewer spatial samples → more aliasing and detail loss"
    )

# Image loading
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
    up = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"], help="High resolution recommended")
    if up:
        img = Image.open(up).convert("RGB")
else:
    if source == "Checkerboard":
        img = checkerboard()
    elif source == "Vertical stripes":
        img = vertical_stripes()

# Interpolation options with better labels
interp_options = {
    "Nearest": "🟦 Nearest (Blocky, fastest)",
    "Bilinear": "🟩 Bilinear (Smooth)",
    "Bicubic": "🟨 Bicubic (Better detail)",
    "Lanczos": "⭐ Lanczos (Sharpest practical)",
    "Sinc (Ideal Approximation)": "📐 Sinc (Theoretical best approx)"
}

interp_name = st.selectbox(
    "🔬 **Reconstruction Method**",
    list(interp_options.keys()),
    format_func=lambda x: interp_options[x]
)

# Helper functions
def mse(a, b):
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    return np.mean((aa - bb) ** 2)

def psnr(m):
    if m == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(m))

interp_map = {
    "Nearest": Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic": Image.Resampling.BICUBIC,
    "Lanczos": Image.Resampling.LANCZOS
}

if img is not None:
    w, h = img.size
    sw = max(1, int(w * sampling / 100))
    sh = max(1, int(h * sampling / 100))

    # Downsample
    sampled = img.resize((sw, sh), Image.Resampling.BOX)

    # Reconstruct
    if interp_name == "Sinc (Ideal Approximation)":
        recon = sampled.resize((w, h), Image.Resampling.LANCZOS)
    else:
        recon = sampled.resize((w, h), interp_map[interp_name])

    m = mse(img, recon)
    p = psnr(m)

    # Image display
    st.markdown("### 📊 Visual Comparison")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(img, caption="**Original**", use_container_width=True)
    with c2:
        st.image(sampled, caption=f"**Sampled** ({sw}×{sh})", use_container_width=True)
    with c3:
        st.image(recon, caption="**Reconstructed**", use_container_width=True)

    st.divider()

    # Metrics
    st.markdown("### 📈 Quality Metrics")
    met1, met2, met3 = st.columns(3)

    with met1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Resolution</h3>
            <h2>{sw} × {sh}</h2>
            <small>from {w}×{h}</small>
        </div>
        """, unsafe_allow_html=True)

    with met2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Mean Squared Error</h3>
            <h2>{m:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with met3:
        psnr_display = "∞" if np.isinf(p) else f"{p:.2f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>PSNR (dB)</h3>
            <h2>{psnr_display}</h2>
            <small>Higher is better</small>
        </div>
        """, unsafe_allow_html=True)

    # Insights
    st.markdown("---")
    insight_col1, insight_col2 = st.columns([2, 1])

    with insight_col1:
        if source != "Upload image":
            if sampling >= 60:
                st.success("✅ **Nyquist condition likely satisfied** — Good reconstruction for this pattern.")
            elif sampling >= 35:
                st.warning("⚠️ **Borderline sampling** — Some aliasing or softening may appear.")
            else:
                st.error("❌ **Severe undersampling** — Strong aliasing and detail loss expected.")
        else:
            st.info("**Note**: Real photos contain many frequencies. Lower sampling always loses fine detail.")

    with insight_col2:
        st.markdown("**Tip**: Try the checkerboard or stripes with low sampling density to clearly see aliasing artifacts.")

    st.markdown("""
    ### 🎯 Key Takeaways
    - The **Nyquist–Shannon theorem** states that to perfectly reconstruct a signal, you must sample at **at least twice** its highest frequency.
    - In images, high-frequency patterns (fine details, sharp edges) require high sampling density.
    - **Lanczos** and **Sinc** give the sharpest results, while **Nearest** produces visible blocks.
    """)

else:
    st.info("👆 Choose a built-in test pattern or upload your own image to begin exploring sampling theory.")

st.caption("Nyquist–Shannon Image Sampling & Reconstruction Simulator • Built with Streamlit")
