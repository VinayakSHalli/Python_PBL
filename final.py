import streamlit as st
import numpy as np
from PIL import Image
import math

# 1. Page Configuration (Always first)
st.set_page_config(
    page_title="Nyquist–Shannon Image Sampling Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to beautify elements, cards, and images
st.markdown("""
    <style>
        /* Main Container Padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        /* Style Streamlit Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Style images with subtle borders and shadows */
        .stImage img {
            border-radius: 8px;
            border: 1px solid #E0E2E6;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        /* Callout styling adjustments */
        .element-container div.stAlert {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def checkerboard(size=512, block=8):
    y, x = np.indices((size, size))
    arr = (((x // block) + (y // block)) % 2) * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

def vertical_stripes(size=512, stripe=4):
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        arr[:, i] = 255 if (i // stripe) % 2 == 0 else 0
    return Image.fromarray(arr).convert("RGB")

def mse(a, b):
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    return np.mean((aa - bb) ** 2)

def psnr(m):
    if m == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(m))

# --- Dictionary for Dynamic Descriptions ---
method_descriptions = {
    "Nearest": "**Nearest Neighbor:** Copies the nearest pixel. Fast and preserves original sharp pixel boundaries, but results in a highly blocky appearance.",
    "Bilinear": "**Bilinear Interpolation:** Averages the 2x2 neighborhood of pixels. Produces smoother results but can introduce significant blurring.",
    "Bicubic": "**Bicubic Interpolation:** Uses a 4x4 neighborhood to calculate weighted pixel values. Preserves fine details better than bilinear.",
    "Lanczos": "**Lanczos Resampling:** Uses a high-quality windowed sinc kernel. Tends to provide the sharpest reconstructions for natural photographic details.",
    "Sinc (Ideal Approximation)": "**Sinc (Ideal Approximation):** Theoretically perfect for band-limited signals under the Nyquist criterion. Approximated here using Lanczos because standard image libraries do not support infinite sinc kernels."
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🎛️ Simulation Controls")
    st.caption("Adjust parameters to see real-time sampling & aliasing effects.")
    st.divider()
    
    source = st.radio(
        "1. Select Image Source",
        ["Upload image", "Checkerboard", "Vertical stripes"]
    )
    
    if source == "Upload image":
        up = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
        img = Image.open(up).convert("RGB") if up else None
    elif source == "Checkerboard":
        img = checkerboard()
    else:
        img = vertical_stripes()
        
    st.divider()
    
    sampling = st.slider("2. Effective Sampling Density (%)", 5, 100, 100, 5)
    
    st.divider()
    
    interp_name = st.selectbox(
        "3. Reconstruction Interpolation",
        list(method_descriptions.keys())
    )
    
    # Dynamic context box showing details on the selected interpolation method
    st.info(method_descriptions[interp_name])

# --- Main App Layout ---
st.title("✨ Nyquist–Shannon Image Sampling
