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

# ====================== STRONGER HIERARCHY CSS ======================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
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
        font-size: 1.6rem;
        margin-bottom: 2.5rem;
    }

    /* === MODE SELECTOR - Largest after title === */
    .stRadio > label {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
    }
    .stRadio div[role="radiogroup"] label {
        font-size: 1.65rem !important;
    }

    /* Section Headers */
    .section-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E40AF;
        margin: 2.5rem 0 1.3rem 0;
    }

    /* Image Labels */
    .image-label {
        font-size: 2.0rem !important;
        font-weight: 600;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.7);
    }

    /* Control Labels - Increased size */
    .stSelectbox label, 
    .stRadio label:not(.stRadio > label), 
    .stSlider label, 
    .stNumberInput label, 
    .stFileUploader label {
        font-size: 1.55rem !important;
        font-weight: 600 !important;
        color: #1F2937;
    }

    /* Metric Labels - Bigger */
    .stMetric label {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    
    /* Metric Values - Prominent */
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 700;
        color: #1E3A8A;
    }

    /* General content */
    p, .stMarkdown, div {
        font-size: 1.18rem;
        line-height: 1.65;
    }

    .stImage img {
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Explore Sampling Theory in Images & Audio Signals</p>', unsafe_allow_html=True)

# Mode Selector
mode = st.radio(
    "Select Domain to Explore",
    ["🖼️ Image Sampling", "🎵 Audio Sampling"],
    horizontal=True
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

        m = np.mean((np.array(img, dtype=np.float32) - np.array(recon, dtype=np.float32)) ** 2)
        p = float("inf") if m == 0 else 20 * math.log10(255.0 / math.sqrt(m))

        st.markdown('<h3 class="section-header">Visual Comparison</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p class="image-label">Original</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with c2:
            st.markdown('<p class="image-label">Sampled</p>', unsafe_allow_html=True)
            st.image(sampled, use_container_width=True)
        with c3:
            st.markdown('<p class="image-label">Reconstructed</p>', unsafe_allow_html=True)
            st.image(recon, use_container_width=True)

        st.divider()
        st.markdown('<h3 class="section-header">Quality Metrics</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Sampled Resolution", f"{sw} × {sh}")
        with m2: st.metric("Mean Squared Error (MSE)", f"{m:.2f}")
        with m3: st.metric("PSNR (dB)", "∞" if np.isinf(p) else f"{p:.2f}")

        if source != "Upload image":
            if sampling >= 60:
                st.success("Nyquist condition likely satisfied — Good reconstruction")
            elif sampling >= 35:
                st.warning("Borderline sampling — Some aliasing may appear")
            else:
                st.error("Severe undersampling — Strong aliasing expected")

# ====================== AUDIO SAMPLING ======================
elif mode == "🎵 Audio Sampling":
    st.markdown('<h3 class="section-header">Audio Signal Sampling</h3>', unsafe_allow_html=True)
    
    st.markdown("### Upload Audio Signal (.wav)")
    uploaded_file = st.file_uploader("", type=["wav"], label_visibility="collapsed")

    def load_audio(file_obj):
        fs, audio = wavfile.read(file_obj)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(float)
        audio = audio / np.max(np.abs(audio))
        return fs, audio

    if uploaded_file:
        fs, audio = load_audio(io.BytesIO(uploaded_file.read()))
    else:
        try:
            fs, audio = load_audio("101_1b1_Pr_sc_Meditron.wav")
        except:
            st.warning("Please upload a .wav file to begin")
            st.stop()

    st.markdown("### Original Signal")
    st.audio((audio * 32767).astype(np.int16), sample_rate=fs)

    # [Rest of your audio code remains the same...]
    fft_vals = np.abs(np.fft.rfft(audio))
    fft_freqs = np.fft.rfftfreq(len(audio), 1/fs)
    dominant_frequency = float(fft_freqs[np.argmax(fft_vals[1:]) + 1])
    threshold = 0.05 * np.max(fft_vals)
    max_frequency = float(np.max(fft_freqs[fft_vals > threshold]))
    nyquist = max(1, int(2 * max_frequency))

    col1, col2 = st.columns([2, 1])
    with col1:
        rate = int(st.number_input("Sampling Rate (Hz)", min_value=1, value=max(1000, nyquist), step=100))
    with col2:
        method = st.selectbox("Reconstruction Method", 
                            ["Nearest Neighbor", "Linear", "Cubic", "Lanczos", "Sinc"])

    # ... (keep the rest of your audio reconstruction, plots, and metrics code as before) ...

st.caption("Nyquist–Shannon Sampling Explorer • Images + Audio")
