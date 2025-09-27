# visual.py
# Streamlit — Image Enhancement with Image Decomposition & Visual Adaptation
# Run: streamlit run visual.py

import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.color import rgb2hsv, hsv2rgb

# =========================
# Page config
# =========================
st.set_page_config(page_title="Decomposition + Visual Adaptation", page_icon="🩺", layout="wide")
st.title("🩺 Image Enhancement: Decomposition + Visual Adaptation")
st.caption("TV-based decomposition → illuminant correction (Naka–Rushton) → detail fusion with noise suppression")

# =========================
# Helpers
# =========================
def to_float01(a):
    """Convert to float32 in [0,1]."""
    a = np.asarray(a)
    if a.dtype.kind in ("u", "i"):
        a = a.astype(np.float32)
        if a.max() > 1:
            a /= 255.0
    else:
        a = a.astype(np.float32)
    return np.clip(a, 0.0, 1.0)

def pil_to_nd(img_pil, mode):
    """PIL -> np.float32 [0,1], with Auto/Grayscale/Color handling."""
    if mode == "Grayscale":
        if img_pil.mode != "L":
            img_pil = img_pil.convert("L")
        return to_float01(np.array(img_pil)), False
    if mode == "Color":
        img_pil = img_pil.convert("RGB")
        return to_float01(np.array(img_pil)), True
    # Auto-detect
    if img_pil.mode in ("L", "I;16", "I", "F"):
        return to_float01(np.array(img_pil)), False
    return to_float01(np.array(img_pil.convert("RGB"))), True

def nd_to_pil(a):
    """np [0,1] -> PIL uint8."""
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        return Image.fromarray((a * 255).astype(np.uint8), mode="L")
    return Image.fromarray((a * 255).astype(np.uint8), mode="RGB")

def show_hist(image, is_color, title):
    """Return a histogram figure."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    if is_color:
        for i, label in enumerate(["R", "G", "B"]):
            ax.hist(image[..., i].ravel(), bins=256, range=(0, 1), histtype="step", linewidth=1.4, label=label)
        ax.legend()
    else:
        ax.hist(image.ravel(), bins=256, range=(0, 1), histtype="stepfilled", alpha=0.85)
    ax.set_title(title); ax.set_xlabel("Intensity (0–1)"); ax.set_ylabel("Count"); fig.tight_layout()
    return fig

# =========================
# Decomposition primitives
# =========================
# Kernel for global noise estimation (Immerkaer-like)
NS_KERNEL = np.array([[1, -2, 1],
                      [-2, 4, -2],
                      [1, -2, 1]], dtype=np.float32)

def estimate_lambda1(channel):
    """Global noise estimation → λ1 scale."""
    H, W = channel.shape
    conv = convolve(channel, NS_KERNEL, mode="nearest")[1:-1, 1:-1]  # trim border affected by kernel
    s = np.abs(conv).sum()
    denom = 6.0 * max(W - 2, 1) * max(H - 2, 1)
    lam = np.sqrt(np.pi / 2.0) * (s / denom)
    return float(lam)

def tv_decompose_per_channel(ch, weight):
    """TV-chambolle denoise to obtain 'structure' (base+detail)."""
    return denoise_tv_chambolle(ch, weight=weight, eps=2e-4)

def decompose_layers(img, is_color, lam1_gain, lam2):
    """
    Two-stage TV decomposition:
    1) img -> structure + noise   (λ1 from global noise * lam1_gain)
    2) structure -> base + detail (λ2 user)
    """
    if is_color:
        structure = np.zeros_like(img)
        noise = np.zeros_like(img)
        for c in range(3):
            lam1 = estimate_lambda1(img[..., c]) * lam1_gain
            s = tv_decompose_per_channel(img[..., c], weight=lam1)
            n = img[..., c] - s
            structure[..., c] = s
            noise[..., c] = n

        base = np.zeros_like(img)
        detail = np.zeros_like(img)
        for c in range(3):
            b = tv_decompose_per_channel(structure[..., c], weight=lam2)
            base[..., c] = b
            detail[..., c] = structure[..., c] - b
    else:
        lam1 = estimate_lambda1(img) * lam1_gain
        structure = tv_decompose_per_channel(img, weight=lam1)
        noise = img - structure
        base = tv_decompose_per_channel(structure, weight=lam2)
        detail = structure - base

    return base, detail, noise

def visual_adaptation_on_base(base, is_color, n_exp=1.0):
    """
    Naka–Rushton visual adaptation on luminance of base.
    Grayscale: apply directly. Color: HSV-V channel.
    """
    if is_color:
        hsv = rgb2hsv(np.clip(base, 0, 1))
        V = hsv[..., 2]
        Mg = float(V.mean()); Sg = float(V.std())
        sigma_g = Mg / (1.0 + np.exp(Sg))
        Lout = (V**n_exp) / ((V**n_exp) + (sigma_g**n_exp) + 1e-9)
        hsv[..., 2] = np.clip(Lout, 0, 1)
        return np.clip(hsv2rgb(hsv), 0, 1)
    X = np.clip(base, 0, 1)
    Mg = float(X.mean()); Sg = float(X.std())
    sigma_g = Mg / (1.0 + np.exp(Sg))
    Lout = (X**n_exp) / ((X**n_exp) + (sigma_g**n_exp) + 1e-9)
    return np.clip(Lout, 0, 1)

def fuse_detail(enh_base, detail, is_color, alpha_r, alpha_g, alpha_b, gauss_sigma):
    """
    Weighted fusion with denoising/artifact suppression.
    - Discard noise layer implicitly.
    - Blue-channel weight often set lower to suppress artifacts.
    """
    if is_color:
        out = enh_base.copy()
        for c, alpha in enumerate([alpha_r, alpha_g, alpha_b]):  # R,G,B
            w = alpha * gaussian_filter(np.abs(detail[..., c]), sigma=gauss_sigma)
            out[..., c] = enh_base[..., c] + w * detail[..., c]
        return np.clip(out, 0, 1)
    # grayscale: reuse alpha_r
    w = alpha_r * gaussian_filter(np.abs(detail), sigma=gauss_sigma)
    out = enh_base + w * detail
    return np.clip(out, 0, 1)

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("1) Upload")
    f = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    mode = st.radio("Interpret upload as", ["Auto-detect", "Grayscale", "Color"], index=0)

    st.header("2) Decomposition")
    lam1_gain = st.slider("λ₁ gain (noise → structure)", 0.1, 5.0, 1.0, 0.1,
                          help="Scale for λ₁ (global-noise-based). Higher = smoother structure, more noise removed.")
    lam2 = st.slider("λ₂ (structure → base)", 0.05, 1.00, 0.30, 0.05,
                     help="TV weight to split base/detail (≈0.3 works well).")

    st.header("3) Visual Adaptation")
    n_exp = st.slider("Exponent n (Naka–Rushton)", 0.50, 2.50, 1.00, 0.05,
                      help="n≈1 per paper. <1 flattens, >1 emphasizes bright regions.")

    st.header("4) Detail Fusion")
    alpha_r = st.slider("α_R (red / or grayscale)", 0.0, 5.0, 2.0, 0.1)
    alpha_g = st.slider("α_G (green)", 0.0, 5.0, 2.0, 0.1)
    alpha_b = st.slider("α_B (blue, artifacts)", 0.0, 2.0, 0.0, 0.1)
    gauss_sigma = st.slider("Gaussian σ for |detail| smoothing", 2, 20, 10, 1)

    st.header("5) Output")
    enable_dl = st.checkbox("Enable download", True)

# =========================
# Guard
# =========================
if f is None:
    st.info("Upload an image to begin.")
    st.stop()

# =========================
# Load & (optional) resize guard
# =========================
pil = Image.open(f)
img, is_color = pil_to_nd(pil, mode)

# (Optional) protect from very large inputs
MAX_SIDE = 2048
h, w = img.shape[:2]
scale = min(1.0, MAX_SIDE / max(h, w))
if scale < 1.0:
    pil_small = nd_to_pil(img).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img, _ = pil_to_nd(pil_small, "Color" if is_color else "Grayscale")

# =========================
# Pipeline
# =========================
base, detail, noise = decompose_layers(img, is_color, lam1_gain=lam1_gain, lam2=lam2)
enh_base = visual_adaptation_on_base(base, is_color=is_color, n_exp=n_exp)
proc = fuse_detail(enh_base, detail, is_color, alpha_r, alpha_g, alpha_b, gauss_sigma)

# =========================
# Display
# =========================
st.subheader("Preview")
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("**Before**")
    st.image(nd_to_pil(img), use_container_width=True)
    st.pyplot(show_hist(img, is_color, "Histogram (Before)"), clear_figure=True)

with c2:
    st.markdown("**After — Decomposition + Visual Adaptation**")
    st.image(nd_to_pil(proc), use_container_width=True)
    st.pyplot(show_hist(proc, is_color, "Histogram (After)"), clear_figure=True)

with st.expander("Layer inspection (optional)"):
    st.markdown("• **Base (after TV split, before adaptation)**")
    st.image(nd_to_pil(base if is_color else base), use_container_width=True)
    st.markdown("• **Detail layer (visualized, normalized)**")
    dvis = (detail - detail.min()) / (detail.max() - detail.min() + 1e-9)
    st.image(nd_to_pil(dvis), use_container_width=True)
    st.markdown("• **Noise layer (visualized, normalized; discarded in fusion)**")
    nvis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    st.image(nd_to_pil(nvis), use_container_width=True)

# =========================
# Download
# =========================
if enable_dl:
    buf = io.BytesIO()
    nd_to_pil(proc).save(buf, format="PNG")
    st.download_button("Download enhanced image", buf.getvalue(),
                       file_name="enhanced_decomposition.png", mime="image/png", type="primary")

st.caption("Implementation note: TV decomposition via Chambolle; visual adaptation on luminance; detail fusion with Gaussian-weighted gain.")
