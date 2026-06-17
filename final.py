
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import math

st.set_page_config(page_title="Nyquist–Shannon Image Sampling Explorer", layout="wide")

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

h1{
    text-align:center;
    color:#0F4C81;
}

div[data-testid="stMetric"]{
    background-color:#F5F7FA;
    border-radius:12px;
    padding:10px;
    border:1px solid #DDDDDD;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 📷 Nyquist–Shannon Image Sampling Explorer

### Interactive demonstration of spatial sampling, reconstruction, and aliasing

---
""")

st.info(
    "💡 This simulator treats a high-resolution image as an approximation of a "
    "continuous scene and demonstrates the effects of changing the sampling density."
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

def mse(a,b):
    aa=np.asarray(a).astype(np.float32)
    bb=np.asarray(b).astype(np.float32)
    return np.mean((aa-bb)**2)

def psnr(m):
    if m == 0:
        return float("inf")
    return 20*math.log10(255.0/math.sqrt(m))

with st.sidebar:

    st.header("⚙️ Controls")

    source = st.radio(
        "Image Source",
        ["Upload image", "Checkerboard", "Vertical stripes"]
    )

    sampling = st.slider(
        "Effective Sampling Density (%)",
        5, 100, 100, 5
    )

    interp_name = st.selectbox(
        "Reconstruction Method",
        [
            "Nearest",
            "Bilinear",
            "Bicubic",
            "Lanczos",
            "Sinc (Ideal Approximation)"
        ]
    )

    st.markdown("---")

    st.caption("""
**Interpolation methods**

• Nearest → Fast but blocky

• Bilinear → Smooth averaging

• Bicubic → Better detail preservation

• Lanczos → High-quality windowed sinc approximation

• Sinc → Demonstrated using Lanczos in this simulator
""")
    
interp_map = {
    "Nearest": Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic": Image.Resampling.BICUBIC,
    "Lanczos": Image.Resampling.LANCZOS
}

if img is not None:
    w,h = img.size
    sw=max(1,int(w*sampling/100))
    sh=max(1,int(h*sampling/100))

    sampled = img.resize((sw,sh), Image.Resampling.BOX)
    if interp_name == "Sinc (Ideal Approximation)":
    # Pillow has no true sinc interpolator.
    # Lanczos uses a windowed sinc kernel and is the closest practical approximation.
        recon = sampled.resize((w, h), Image.Resampling.LANCZOS)
    else:
        recon = sampled.resize((w, h), interp_map[interp_name])

    m = mse(img,recon)
    p = psnr(m)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("Original Image")
        st.image(img,use_container_width=True)
    with c2:
        st.subheader("Sampled")
        st.image(sampled,use_container_width=True)
    with c3:
        st.subheader("Reconstructed")
        st.image(recon,use_container_width=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📏 Sampled Resolution", f"{sw} × {sh}")
    with m2:
        st.metric("📉 MSE", f"{m:.2f}")
    with m3:
        st.metric(
            "📊 PSNR",
            "∞" if np.isinf(p) else f"{p:.2f} dB"
        )

    if source != "Upload image":

        if sampling >= 50:
            st.success(
            "✅ The sampling density is relatively high for this synthetic pattern, so fine details are more likely to be reconstructed accurately."
            )
        else:
            st.error(
                "⚠️ The sampling density is low. Aliasing and irreversible loss of detail may occur because the Nyquist criterion may not be satisfied."
                )

    else:
        st.info(
            "📸 Uploaded photographs have unknown maximum spatial frequencies. "
            "Reducing the sampling density generally removes fine details and can introduce aliasing."
            )

    st.markdown("""
### Interpretation
- **Original image:** Used as a high-resolution stand-in for a continuous scene.
- **Sampled image:** Represents fewer spatial measurements.
- **Reconstructed image:** Upscaled estimate built from those samples.

For synthetic checkerboards and stripes, reducing the sampling density can violate the
Nyquist criterion for their spatial frequencies, producing visible artifacts.
""")

    with st.expander("📘 Learn about the Nyquist–Shannon Sampling Theorem"):
        st.markdown("""
- Real-world scenes are continuous.
- Digital images store discrete pixel samples.
- If sampling density is too low, high-frequency details cannot be reconstructed correctly.
- This leads to **aliasing**, distortion, or loss of detail.
- Synthetic patterns like checkerboards and stripes clearly demonstrate these effects.
""")
else:
    st.info("Choose a built-in pattern or upload an image.")
    
st.markdown("---")

st.caption(
    "Developed as a Python Project-Based Learning demonstration of the Nyquist–Shannon Sampling Theorem."
)
