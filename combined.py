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

st.markdown("""
<style>
    /* ── Type scale ──────────────────────────────────────────────
       h1  : 2.2rem  page title
       h2  : 1.5rem  section headers
       base: 1.0rem  body / labels / captions
       sm  : 0.875rem  helper text
    ─────────────────────────────────────────────────────────── */

    .main .block-container { padding-top: 2rem; }

    /* Page title */
    .pg-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }

    /* Page subtitle */
    .pg-subtitle {
        font-size: 1.0rem;
        font-weight: 400;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .sec-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 2rem 0 0.75rem;
        padding-left: 0.75rem;
        border-left: 4px solid #3B82F6;
    }

    /* Image column labels */
    .img-label {
        font-size: 1.0rem;
        font-weight: 600;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* ── Streamlit widget labels ─────────────────────────────── */
    /* Radio group outer label */
    .stRadio > label {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    /* Radio option text */
    .stRadio div[role="radiogroup"] label span {
        font-size: 1.0rem !important;
    }
    /* Slider, selectbox, file uploader, number input labels */
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stNumberInput"] label {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    div[data-testid="stMetric"] label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }

    /* ── Body text ───────────────────────────────────────────── */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.0rem !important;
        line-height: 1.6;
        color: #374151;
    }

    /* Caption */
    div[data-testid="stCaptionContainer"] {
        font-size: 0.875rem !important;
        color: #94A3B8 !important;
    }

    /* Alert messages */
    div[data-testid="stAlert"] {
        font-size: 1.0rem !important;
    }

    /* Image rounding */
    .stImage img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="pg-title">Nyquist–Shannon Sampling Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="pg-subtitle">Explore sampling theory in images & audio signals</p>', unsafe_allow_html=True)

# ── Mode selector ──────────────────────────────────────────────────────
mode = st.radio(
    "Select domain to explore",
    ["🖼️ Image Sampling", "🎵 Audio Sampling"],
    horizontal=True
)
st.divider()

# ── Image Sampling ─────────────────────────────────────────────────────
if mode == "🖼️ Image Sampling":
    st.markdown('<h2 class="sec-header">How sampling works</h2>', unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**1. Original image** — high-resolution reference")
    with w2:
        st.markdown("**2. Sampling** — downsample (fewer spatial samples)")
    with w3:
        st.markdown("**3. Reconstruction** — upscale with interpolation")
    st.divider()

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        source = st.radio(
            "Image source",
            ["Upload image", "Checkerboard", "Vertical stripes"],
            horizontal=True
        )
    with col_ctrl2:
        sampling = st.slider(
            "Effective sampling density (%)",
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
    elif source == "Checkerboard":
        img = checkerboard()
    elif source == "Vertical stripes":
        img = vertical_stripes()

    interp_options = {
        "Nearest": "Nearest (blocky)",
        "Bilinear": "Bilinear (smooth)",
        "Bicubic": "Bicubic (good detail)",
        "Lanczos": "Lanczos (sharpest)",
        "Sinc (Ideal Approximation)": "Sinc (theoretical best)",
    }
    interp_name = st.selectbox(
        "Reconstruction method",
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
                "Lanczos": Image.Resampling.LANCZOS,
            }
            recon = sampled.resize((w, h), interp_map[interp_name])

        m = np.mean((np.array(img, dtype=np.float32) - np.array(recon, dtype=np.float32)) ** 2)
        p = float("inf") if m == 0 else 20 * math.log10(255.0 / math.sqrt(m))

        st.markdown('<h2 class="sec-header">Visual comparison</h2>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p class="img-label">Original</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with c2:
            st.markdown('<p class="img-label">Sampled</p>', unsafe_allow_html=True)
            st.image(sampled, use_container_width=True)
        with c3:
            st.markdown('<p class="img-label">Reconstructed</p>', unsafe_allow_html=True)
            st.image(recon, use_container_width=True)

        st.divider()
        st.markdown('<h2 class="sec-header">Quality metrics</h2>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Sampled resolution", f"{sw} × {sh}")
        with m2:
            st.metric("Mean squared error (MSE)", f"{m:.2f}")
        with m3:
            st.metric("PSNR (dB)", "∞" if np.isinf(p) else f"{p:.2f}")

        if source != "Upload image":
            if sampling >= 60:
                st.success("Nyquist condition likely satisfied — good reconstruction")
            elif sampling >= 35:
                st.warning("Borderline sampling — some aliasing may appear")
            else:
                st.error("Severe undersampling — strong aliasing expected")

# ── Audio Sampling ─────────────────────────────────────────────────────
elif mode == "🎵 Audio Sampling":
    st.markdown('<h2 class="sec-header">Audio signal sampling</h2>', unsafe_allow_html=True)
    st.markdown("Upload a WAV file to get started.")

    uploaded_file = st.file_uploader("Audio file (.wav)", type=["wav"])

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
            st.warning("Please upload a .wav file to begin.")
            st.stop()

    st.markdown("**Original signal**")
    st.audio((audio * 32767).astype(np.int16), sample_rate=fs)

    fft_vals = np.abs(np.fft.rfft(audio))
    fft_freqs = np.fft.rfftfreq(len(audio), 1 / fs)
    dominant_frequency = float(fft_freqs[np.argmax(fft_vals[1:]) + 1])
    threshold = 0.05 * np.max(fft_vals)
    max_frequency = float(np.max(fft_freqs[fft_vals > threshold]))
    nyquist = max(1, int(2 * max_frequency))

    col1, col2 = st.columns([2, 1])
    with col1:
        rate = int(st.number_input(
            "Sampling rate (Hz)",
            min_value=1, value=max(1000, nyquist), step=100
        ))
    with col2:
        method = st.selectbox(
            "Reconstruction method",
            ["Nearest Neighbor", "Linear", "Cubic", "Lanczos", "Sinc"]
        )

    step = max(1, int(fs / rate))
    sampled = audio[::step]
    sample_pos = np.arange(0, len(sampled) * step, step)
    full_pos = np.arange(len(audio))

    def lanczos_kernel(x, a=3):
        x = np.array(x, dtype=float)
        out = np.sinc(x) * np.sinc(x / a)
        out[np.abs(x) >= a] = 0
        return out

    def lanczos_resample(sample_pos, sampled, full_pos, a=3):
        reconstructed = np.zeros(len(full_pos))
        for i, t in enumerate(full_pos):
            x = (t - sample_pos) / np.mean(np.diff(sample_pos))
            weights = lanczos_kernel(x, a)
            s = np.sum(weights)
            if abs(s) > 1e-12:
                reconstructed[i] = np.sum(sampled * weights) / s
        return reconstructed

    def sinc_resample(sample_pos, sampled, full_pos, window=8):
        reconstructed = np.zeros(len(full_pos))
        Ts = np.mean(np.diff(sample_pos))
        for i, t in enumerate(full_pos):
            x = (t - sample_pos) / Ts
            mask = np.abs(x) <= window
            x_local = x[mask]
            samples_local = sampled[mask]
            weights = np.sinc(x_local)
            hamming = 0.54 + 0.46 * np.cos(np.pi * x_local / window)
            weights *= hamming
            s = np.sum(weights)
            if abs(s) > 1e-12:
                reconstructed[i] = np.sum(samples_local * weights) / s
        return reconstructed

    if method == "Nearest Neighbor":
        reconstructed = interp1d(sample_pos, sampled, kind="nearest", fill_value="extrapolate")(full_pos)
    elif method == "Linear":
        reconstructed = interp1d(sample_pos, sampled, kind="linear", fill_value="extrapolate")(full_pos)
    elif method == "Cubic":
        reconstructed = interp1d(sample_pos, sampled, kind="cubic", fill_value="extrapolate")(full_pos)
    elif method == "Lanczos":
        reconstructed = lanczos_resample(sample_pos, sampled, full_pos)
    elif method == "Sinc":
        reconstructed = sinc_resample(sample_pos, sampled, full_pos)

    reconstructed_audio = reconstructed.copy()
    max_amp = np.max(np.abs(reconstructed_audio))
    if max_amp > 0:
        reconstructed_audio = reconstructed_audio / max_amp

    st.markdown("**Reconstructed signal**")
    st.audio((reconstructed_audio * 32767).astype(np.int16), sample_rate=fs)

    error = audio - reconstructed
    accuracy = max(0, min(100, 100 - np.mean(np.abs(error)) * 100))

    st.markdown('<h2 class="sec-header">Signal metrics</h2>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dominant frequency", f"{dominant_frequency:.2f} Hz")
    m2.metric("Nyquist rate", f"{nyquist} Hz")
    m3.metric("Sampling rate", f"{rate} Hz")
    m4.metric("Accuracy", f"{accuracy:.2f}%")

    display = min(4000, len(audio))
    x = np.arange(display)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=audio[:display], name="Original",
        line=dict(color="#1E3A8A", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=reconstructed[:display], name="Reconstructed",
        line=dict(color="#B45309", width=2)
    ))
    fig.update_layout(
        title="Original vs reconstructed signal (first 4000 samples)",
        height=420,
        title_font_size=14,
        legend_font_size=13,
        font_size=13,
        xaxis_title="Sample index",
        yaxis_title="Amplitude",
        margin=dict(t=48, b=40, l=48, r=16),
    )
    st.plotly_chart(fig, use_container_width=True)

    if rate < nyquist:
        st.error("Aliasing detected — sampling rate is below the Nyquist rate.")
    elif rate < 1.5 * nyquist:
        st.warning("Acceptable reconstruction — consider raising the sampling rate.")
    else:
        st.success("High-fidelity reconstruction.")

st.caption("Nyquist–Shannon Sampling Explorer · Images + Audio")
