# app_decomposition.py
# Streamlit — Image Enhancement with Image Decomposition & Visual Adaptation
# Ref: Wang et al. 2021 (fundus enhancement via base/detail/noise + adaptation)
# Run: streamlit run app_decomposition.py

import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.color import rgb2hsv, hsv2rgb

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Decomposition + Visual Adaptation", page_icon="🩺", layout="wide")
st.title("🩺 Image Enhancement: Decomposition + Visual Adaptation")
st.caption("TV-based decomposition → illuminant correction (Naka–Rushton) → detail fusion with noise suppression.")

# -----------------------------
# Helpers
# -----------------------------
def to_float01(a):
    a = np.asarray(a)
    if a.dtype.kind in ("u", "i"):
        a = a.astype(np.float32)
        if a.max() > 1:
            a /= 255.0
    else:
        a = a.astype(np.float32)
    return np.clip(a, 0.0, 1.0)

def pil_to_nd(img_pil, color_mode):
    if color_mode == "Grayscale":
        if img_pil.mode != "L":
            img_pil = img_pil.convert("L")
        arr = to_float01(np.array(img_pil))
        return arr, False
    elif color_mode == "Color":
        img_pil = img_pil.convert("RGB")
        arr = to_float01(np.array(img_pil))
        return arr, True
    else:  # Auto
        if img_pil.mode in ("L", "I;16", "I", "F"):
            arr = to_float01(np.array(img_pil))
            return arr, False
        img_pil = img_pil.convert("RGB")
        arr = to_float01(np.array(img_pil))
        return arr, True

def nd_to_pil(a):
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        return Image.fromarray((a * 255).astype(np.uint8), mode="L")
    return Image.fromarray((a * 255).astype(np.uint8), mode="RGB")

def show_hist(image, is_color, title):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    if is_color:
        for i, label in enumerate(["R", "G", "B"]):
            ax.hist(image[..., i].ravel(), bins=256, range=(0, 1), histtype="step", linewidth=1.4, label=label)
        ax.legend()
    else:
        ax.hist(image.ravel(), bins=256, range=(0, 1), histtype="stepfilled", alpha=0.85)
    ax.set_title(title); ax.set_xlabel("Intensity (0–1)"); ax.set_ylabel("Count"); fig.tight_layout()
    return fig

# ---------- Decomposition primitives (approximation of paper) ----------
# Eq. (5) kernel for global noise estimation
NS_KERNEL = np.array([[1, -2, 1],
                      [-2, 4, -2],
                      [1, -2, 1]], dtype=np.float32)

def estimate_lambda1(channel):
    """Global noise estimation (Immerkaer-like) → λ1 (Eq.4)."""
    H, W = channel.shape
    conv = convolve(channel, NS_KERNEL, mode="nearest")[1:-1, 1:-1]  # valid region
    s = np.abs(conv).sum()
    lam = np.sqrt(np.pi/2.0) * (s / (6.0 * max(W-2,1) * max(H-2,1)))
    return float(lam)

def tv_decompose_per_channel(ch, weight):
    """TV denoise to get 'structure' (base+detail)."""
    # skimage's weight scale is not identical to λ in the paper; we allow a gain knob.
    return denoise_tv_chambolle(ch, weight=weight, eps=2e-4)

def decompose_layers(img, is_color, lam1_gain, lam2):
    """
    Two-stage TV decomposition (Paper Sec.2.1):
    1) separate structure vs noise (λ1 from global noise, scaled by lam1_gain)
    2) split structure → base + detail (λ2 ~ 0.3 in paper)
    Returns: base, detail, noise
    """
    if is_color:
        structure = np.zeros_like(img); noise = np.zeros_like(img)
        for c in range(3):
            lam1 = estimate_lambda1(img[..., c]) * lam1_gain
            s = tv_decompose_per_channel(img[..., c], weight=lam1)
            n = img[..., c] - s
            structure[..., c] = s; noise[..., c] = n
        # second TV to get base from structure
        base = np.zeros_like(img); detail = np.zeros_like(img)
        for c in range(3):
            b = tv_decompose_per_channel(structur_
