# app.py
# Streamlit Image Processing Playground (professional & optimized)
# Run: streamlit run app.py

import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from skimage import exposure, img_as_float
from skimage.color import rgb2hsv, hsv2rgb, rgb2ycbcr, ycbcr2rgb, rgb2gray

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Image Processing Lab",
    page_icon="🎛️",
    layout="wide"
)

# -----------------------------
# Helper utilities
# -----------------------------
def to_float01(arr):
    """Ensure array is float32 in [0,1]."""
    a = np.asarray(arr)
    if a.dtype.kind in ("u", "i"):
        # assume 8-bit if max > 1
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a /= 255.0
    else:
        a = a.astype(np.float32)
    # clip just in case
    return np.clip(a, 0.0, 1.0)

def pil_to_nd(arr_pil, force_gray=False):
    """PIL -> np.float32 [0,1], grayscale or color."""
    if force_gray:
        if arr_pil.mode != "L":
            arr_pil = arr_pil.convert("L")
        a = to_float01(np.array(arr_pil))  # (H, W)
    else:
        if arr_pil.mode in ("L", "I;16", "I", "F"):
            # keep grayscale
            a = to_float01(np.array(arr_pil))
        else:
            arr_pil = arr_pil.convert("RGB")
            a = to_float01(np.array(arr_pil))  # (H, W, 3)
    return a

def nd_to_pil(a):
    """np [0,1] -> PIL uint8."""
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        return Image.fromarray((a * 255).astype(np.uint8), mode="L")
    elif a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray((a * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError("Unsupported image shape for PIL conversion.")

def plot_histogram(image, is_color: bool, title: str):
    """Return a matplotlib figure with histogram(s)."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=150)
    if is_color:
        # per-channel hist (RGB)
        for i, label in enumerate(["R", "G", "B"]):
            ch = image[..., i].ravel()
            ax.hist(ch, bins=256, range=(0, 1), histtype="step", linewidth=1.5, label=label)
        ax.legend()
    else:
        ax.hist(image.ravel(), bins=256, range=(0, 1), histtype="stepfilled", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Intensity (0–1)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def piecewise_linear_transform(img, x1, y1, x2, y2):
    """
    Piecewise linear with 3 segments:
    (0,0)->(x1,y1)->(x2,y2)->(1,1).
    Vectorized via np.interp on flattened array.
    """
    x_points = np.array([0.0, x1, x2, 1.0], dtype=np.float32)
    y_points = np.array([0.0, y1, y2, 1.0], dtype=np.float32)
    x_points = np.clip(x_points, 0, 1)
    y_points = np.clip(y_points, 0, 1)
    # enforce monotonic x to avoid interp errors
    x_points = np.maximum.accumulate(x_points)
    flat = img.reshape(-1)
    out = np.interp(flat, x_points, y_points).astype(np.float32)
    return out.reshape(img.shape)

def apply_hist_equalization_color_rgb(img_rgb, method="global", clip_limit=0.01, tile_grid=(8, 8)):
    """
    Equalize luminance to avoid color shifts.
    method: "global" (HE), "ahe" (clip_limit≈1.0), "clahe" (clip_limit user)
    Uses HSV V channel for global HE and Y in YCbCr for AHE/CLAHE via skimage.adapthist.
    """
    if method == "global":
        hsv = rgb2hsv(img_rgb)
        v = hsv[..., 2]
        v_eq = exposure.equalize_hist(v)
        hsv[..., 2] = np.clip(v_eq, 0, 1)
        return hsv2rgb(hsv)

    # For adaptive methods, using Y (luma) is often a bit more stable
    ycbcr = rgb2ycbcr(img_rgb)
    Y = ycbcr[..., 0] / 255.0  # skimage returns Y in [0,255] float

    if method == "ahe":
        # AHE approximated by equalize_adapthist with clip_limit ~ 1.0 (no clipping),
        # skimage requires (0,1], so we use a high value
        Y_eq = exposure.equalize_adapthist(Y, clip_limit=1.0, nbins=256, kernel_size=tile_grid)
    elif method == "clahe":
        Y_eq = exposure.equalize_adapthist(Y, clip_limit=clip_limit, nbins=256, kernel_size=tile_grid)
    else:
        raise ValueError("Unknown method for color equalization.")

    ycbcr[..., 0] = np.clip(Y_eq * 255.0, 0, 255)
    out = ycbcr2rgb(ycbcr)
    return np.clip(out, 0, 1)

# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.title("🎛️ Image Processing Lab")
st.caption("Select a method, tune parameters, and compare before/after with histograms.")

with st.sidebar:
    st.header("1) Upload Image")
    uploaded = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    img_mode = st.radio("Interpret upload as:", ["Auto-detect", "Grayscale", "Color"], index=0)
    st.divider()
    st.header("2) Choose Method")
    method = st.selectbox(
        "Processing method",
        [
            "Linear Negative",
            "Contrast Stretching",
            "Piecewise Linear Transformation",
            "Log Transformation",
            "Gamma Transformation",
            "Histogram Equalization",
            "Adaptive Histogram Equalization",
            "CLAHE (Contrast Limited AHE)",
        ],
        index=0
    )

    # Per-method parameter controls
    params = {}
    if method == "Contrast Stretching":
        params["low_pct"] = st.slider("Low percentile (%)", 0.0, 10.0, 2.0, 0.1)
        params["high_pct"] = st.slider("High percentile (%)", 90.0, 100.0, 98.0, 0.1)
    elif method == "Piecewise Linear Transformation":
        st.caption("Three segments: (0,0)→(x1,y1)→(x2,y2)→(1,1)")
        params["x1"] = st.slider("x1", 0.0, 1.0, 0.25, 0.01)
        params["y1"] = st.slider("y1", 0.0, 1.0, 0.15, 0.01)
        params["x2"] = st.slider("x2", 0.0, 1.0, 0.75, 0.01)
        params["y2"] = st.slider("y2", 0.0, 1.0, 0.85, 0.01)
    elif method == "Log Transformation":
        st.caption("s = c·log(1 + r); higher c amplifies dark regions.")
        params["c"] = st.slider("c (gain)", 0.1, 5.0, 1.0, 0.1)
    elif method == "Gamma Transformation":
        st.caption("s = r^γ (γ<1 brightens, γ>1 darkens)")
        params["gamma"] = st.slider("γ (gamma)", 0.10, 5.00, 0.80, 0.01)
    elif method == "Adaptive Histogram Equalization":
        params["tile"] = st.slider("Tile grid size", 4, 32, 8, 1)
        st.caption("No clipping (pure AHE). Larger tile increases local contrast.")
    elif method == "CLAHE (Contrast Limited AHE)":
        params["tile"] = st.slider("Tile grid size", 4, 32, 8, 1)
        params["clip"] = st.slider("Clip limit", 0.005, 0.2, 0.01, 0.005)
        st.caption("Clip limit prevents over-amplification of noise.")

    st.divider()
    st.header("3) Output")
    allow_download = st.checkbox("Enable download of processed image", value=True)

# -----------------------------
# Load image
# -----------------------------
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Read with PIL
pil_image = Image.open(uploaded)
force_gray = (img_mode == "Grayscale")
force_color = (img_mode == "Color")

if force_gray:
    np_img = pil_to_nd(pil_image, force_gray=True)  # (H,W)
    is_color = False
elif force_color:
    np_img = pil_to_nd(pil_image, force_gray=False)
    if np_img.ndim == 2:
        # convert to 3-channel grayscale
        np_img = np.dstack([np_img]*3)
    is_color = True
else:
    # auto-detect
    if pil_image.mode in ("L", "I;16", "I", "F"):
        np_img = pil_to_nd(pil_image, force_gray=True)
        is_color = False
    else:
        np_img = pil_to_nd(pil_image, force_gray=False)
        if np_img.ndim == 2:
            np_img = np.dstack([np_img]*3)
            is_color = True
        else:
            is_color = (np_img.ndim == 3 and np_img.shape[2] == 3)

# -----------------------------
# Apply processing
# -----------------------------
proc = None

if method == "Linear Negative":
    # s = 1 - r
    proc = 1.0 - np_img

elif method == "Contrast Stretching":
    low = np.percentile(np_img, params.get("low_pct", 2.0))
    high = np.percentile(np_img, params.get("high_pct", 98.0))
    if is_color:
        # Rescale per channel for stability
        proc = np.empty_like(np_img)
        for c in range(3):
            c_low = np.percentile(np_img[..., c], params["low_pct"])
            c_high = np.percentile(np_img[..., c], params["high_pct"])
            proc[..., c] = exposure.rescale_intensity(np_img[..., c], in_range=(c_low, c_high), out_range=(0, 1))
    else:
        proc = exposure.rescale_intensity(np_img, in_range=(low, high), out_range=(0, 1))

elif method == "Piecewise Linear Transformation":
    x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
    if is_color:
        proc = np.empty_like(np_img)
        for c in range(3):
            proc[..., c] = piecewise_linear_transform(np_img[..., c], x1, y1, x2, y2)
    else:
        proc = piecewise_linear_transform(np_img, x1, y1, x2, y2)

elif method == "Log Transformation":
    c = params["c"]
    # Normalize after log to [0,1]
    if is_color:
        proc = c * np.log1p(np_img)
        proc -= proc.min()
        proc /= (proc.max() + 1e-9)
    else:
        proc = c * np.log1p(np_img)
        proc -= proc.min()
        proc /= (proc.max() + 1e-9)

elif method == "Gamma Transformation":
    gamma = params["gamma"]
    # s = r^gamma (assuming r in [0,1])
    proc = np.power(np_img, gamma, dtype=np.float32)

elif method == "Histogram Equalization":
    if is_color:
        proc = apply_hist_equalization_color_rgb(np_img, method="global")
    else:
        proc = exposure.equalize_hist(np_img)

elif method == "Adaptive Histogram Equalization":
    # AHE via equalize_adapthist with clip_limit ~ 1.0
    tile = params["tile"]
    kernel = (tile, tile)
    if is_color:
        proc = apply_hist_equalization_color_rgb(np_img, method="ahe", tile_grid=kernel)
    else:
        proc = exposure.equalize_adapthist(np_img, kernel_size=kernel, clip_limit=1.0, nbins=256)

elif method == "CLAHE (Contrast Limited AHE)":
    tile = params["tile"]
    clip = params["clip"]
    kernel = (tile, tile)
    if is_color:
        proc = apply_hist_equalization_color_rgb(np_img, method="clahe", clip_limit=clip, tile_grid=kernel)
    else:
        proc = exposure.equalize_adapthist(np_img, kernel_size=kernel, clip_limit=clip, nbins=256)

else:
    st.error("Unknown method.")
    st.stop()

proc = np.clip(proc, 0.0, 1.0)

# -----------------------------
# Display: before / after + histograms
# -----------------------------
st.subheader("Preview")
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("**Before**")
    st.image(nd_to_pil(np_img), use_column_width=True)
    fig1 = plot_histogram(np_img if not isinstance(np_img, Image.Image) else np.array(np_img), is_color, "Histogram (Before)")
    st.pyplot(fig1, clear_figure=True)

with c2:
    st.markdown("**After — {}**".format(method))
    st.image(nd_to_pil(proc), use_column_width=True)
    fig2 = plot_histogram(proc, is_color, "Histogram (After)")
    st.pyplot(fig2, clear_figure=True)

# -----------------------------
# Download
# -----------------------------
if allow_download:
    out_pil = nd_to_pil(proc)
    buf = io.BytesIO()
    fmt = "PNG"
    out_pil.save(buf, format=fmt)
    st.download_button(
        label="Download processed image",
        data=buf.getvalue(),
        file_name=f"processed_{method.replace(' ', '_').lower()}.png",
        mime="image/png",
        type="primary"
    )

# -----------------------------
# Footer note
# -----------------------------
with st.expander("Notes & Tips"):
    st.markdown(
        """
- **Grayscale vs Color**: Histogram Equalization/AHE/CLAHE on color images works on luminance to avoid hue shifts.
- **Contrast Stretching**: Use percentiles to robustly ignore outliers.
- **Piecewise Linear**: Tune `(x1,y1)` and `(x2,y2)` to lift shadows or compress highlights.
- **Log/Gamma**: Great for bringing out details in dark or bright regions respectively.
        """
    )
