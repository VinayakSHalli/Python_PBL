
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import math

st.set_page_config(page_title="Nyquist–Shannon Image Sampling Explorer", layout="wide")

st.title("Nyquist–Shannon Image Sampling & Reconstruction Simulator")

st.markdown("""
**Workflow**
1. Original image (reference)
2. Downsample (simulate fewer spatial samples)
3. Reconstruct by upsampling
4. Compare quality and observe detail loss / aliasing
""")

def checkerboard(size=512, block=8):
    y, x = np.indices((size, size))
    arr = (((x // block) + (y // block)) % 2) * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

def vertical_stripes(size=512, stripe=4):
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        arr[:, i] = 255 if (i // stripe) % 2 == 0 else 0
    return Image.fromarray(arr).convert("RGB")

def sine_grating(size=512, cycles=40):
    x = np.linspace(0, 2*np.pi*cycles, size)
    row = ((np.sin(x)+1)/2*255).astype(np.uint8)
    arr = np.tile(row, (size,1))
    return Image.fromarray(arr).convert("RGB")

def mse(a,b):
    aa=np.asarray(a).astype(np.float32)
    bb=np.asarray(b).astype(np.float32)
    return np.mean((aa-bb)**2)

def psnr(m):
    if m == 0:
        return float("inf")
    return 20*math.log10(255.0/math.sqrt(m))

source = st.radio(
    "Image source",
    ["Upload image","Checkerboard","Vertical stripes","Sine grating"],
    horizontal=True
)

img = None
if source == "Upload image":
    up = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])
    if up:
        img = Image.open(up).convert("RGB")
else:
    if source == "Checkerboard":
        img = checkerboard()
    elif source == "Vertical stripes":
        img = vertical_stripes()
    else:
        img = sine_grating()

sampling = st.slider("Effective Sampling Density (%)", 5, 100, 100, 5)

### Reconstruction Methods
#- **Nearest:** Copies the nearest pixel. Fast but blocky.
#- **Bilinear:** Averages nearby pixels for smoother results.
#- **Bicubic:** Uses more neighbors to preserve detail better.
#- **Lanczos:** Uses a windowed sinc kernel and often produces the sharpest reconstruction.
#- **Sinc (Ideal Approximation):** Theoretically perfect for band-limited signals under the Nyquist criterion. 
#In this demo, it is approximated using Lanczos because standard image libraries do not provide an exact infinite sinc interpolator.

interp_name = st.selectbox(
    "Reconstruction interpolation",
    [
        "Nearest",
        "Bilinear",
        "Bicubic",
        "Lanczos",
        "Sinc (Ideal Approximation)"
    ]
)

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
        st.subheader("Original")
        st.image(img,use_container_width=True)
    with c2:
        st.subheader("Sampled")
        st.image(sampled,use_container_width=True)
    with c3:
        st.subheader("Reconstructed")
        st.image(recon,use_container_width=True)

    st.markdown("---")
    st.metric("Sampled Resolution", f"{sw} × {sh}")
    st.metric("Mean Squared Error (MSE)", f"{m:.2f}")
    st.metric("PSNR (dB)", "∞" if np.isinf(p) else f"{p:.2f}")

    if source != "Upload image":
        if sampling >= 50:
            st.success("Approximate Nyquist condition: likely sufficient for this synthetic pattern.")
        else:
            st.error("Sampling density is low. Aliasing may become visible for this high-frequency pattern.")
    else:
        st.info(
            "For uploaded photographs, the exact highest spatial frequency is unknown. "
            "Lower sampling density generally removes fine detail and can introduce aliasing."
        )

    st.markdown("""
### Interpretation
- **Original image:** Used as a high-resolution stand-in for a continuous scene.
- **Sampled image:** Represents fewer spatial measurements.
- **Reconstructed image:** Upscaled estimate built from those samples.

For synthetic checkerboards and stripes, reducing the sampling density can violate the
Nyquist criterion for their spatial frequencies, producing visible artifacts.
""")
else:
    st.info("Choose a built-in pattern or upload an image.")
