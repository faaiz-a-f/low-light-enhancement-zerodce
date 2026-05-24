import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import time
import os
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry — add or remove versions here
# Place .pth files in the same folder as app.py
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "v4 — SE-Block, per-patch exposure (latest)": {
        "file":    "best_zerodce_seblock_v4.pth",
        "desc":    "No L_tv_enh, per-patch MSE exposure, collapse guard. Best overall quality.",
        "tag":     "Recommended",
        "tag_color": "#1a3a2a",
        "tag_text":  "#68d391",
    },
    "v2 — SE-Block baseline (stable)": {
        "file":    "best_zerodce_seblock_v2.pth",
        "desc":    "Fixed alpha (tanh), correct loss weights, fixed dataset pairing.",
        "tag":     "Stable",
        "tag_color": "#1e3a5f",
        "tag_text":  "#90cdf4",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Architecture (identical to training code)
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DCENet(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(3,  32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        if use_attention:
            self.se = SEBlock(32, reduction=8)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 24, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        nn.init.normal_(self.conv7.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv7.bias, 0)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        if self.use_attention:
            x3 = self.se(x3)
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x4, x3], 1)))
        x6 = self.relu(self.conv6(torch.cat([x5, x2], 1)))
        return torch.tanh(self.conv7(torch.cat([x6, x1], 1)))


def enhance_image_with_curves(x, alpha, iterations=8):
    enhanced = x.clone()
    for i in range(iterations):
        a_i = alpha[:, i*3:(i+1)*3, :, :]
        enhanced = enhanced + a_i * enhanced * (1 - enhanced)
    return torch.clamp(enhanced, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DCENet(use_attention=True).to(device)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    return m, device


def pil_to_tensor(img: Image.Image, device) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def bilateral_denoise(img: Image.Image, d=9, sigma_color=75, sigma_space=75) -> Image.Image:
    """Edge-preserving bilateral filter. Reduces noise without blurring edges."""
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    bgr = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def run_enhance(model, device, img: Image.Image, iterations: int,
                denoise: bool, d: int, sigma_color: int, sigma_space: int):
    tensor   = pil_to_tensor(img, device)
    with torch.no_grad():
        alpha    = model(tensor)
        enhanced = enhance_image_with_curves(tensor, alpha, iterations=iterations)
    result = tensor_to_pil(enhanced)
    if denoise:
        result = bilateral_denoise(result, d=d,
                                   sigma_color=sigma_color,
                                   sigma_space=sigma_space)
    return result


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def avg_brightness(img: Image.Image) -> float:
    return float(np.array(img).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Zero-DCE++ | Low-Light Enhancer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #080b14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, #0d1f3c 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 80% 80%, #0a1628 0%, transparent 60%);
}

[data-testid="stSidebar"] {
    background: #0c1220;
    border-right: 1px solid #1a2540;
}

/* ── version selector cards ── */
.ver-card {
    background: #0f1929;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color .2s;
}
.ver-card:hover   { border-color: #3b6fd4; }
.ver-card.selected{ border-color: #4f8ef7; background: #102040; }
.ver-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.ver-name {
    font-size: 13px;
    font-weight: 700;
    color: #c8d8f0;
    margin-bottom: 4px;
}
.ver-desc { font-size: 11px; color: #4a6080; line-height: 1.5; }

/* ── stat boxes ── */
.stat-box {
    background: #0f1929;
    border: 1px solid #1e3050;
    border-radius: 10px;
    min-height: 88px;
    padding: 14px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.stat-val { font-size: 20px; font-weight: 700; color: #c8d8f0;
            font-family: 'DM Mono', monospace; line-height: 1.1; white-space: nowrap; }
.stat-lbl { font-size: 10px; color: #3a5070; text-transform: uppercase;
            letter-spacing: 1px; margin-top: 3px; }

/* ── img card labels ── */
.img-label {
    font-size: 11px;
    font-weight: 700;
    color: #3a6090;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
}

/* ── enhance button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a4fd6 0%, #2563eb 50%, #1d4ed8 100%);
    color: #e8f0ff;
    border: 1px solid #3b6fd4;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.5px;
    width: 100%;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    border-color: #60a5fa;
}

/* ── download button ── */
.stDownloadButton > button {
    background: #0f1929 !important;
    color: #60a5fa !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    width: 100% !important;
    margin-top: 8px;
}
.stDownloadButton > button:hover {
    border-color: #3b82f6 !important;
    background: #102040 !important;
}

/* ── divider ── */
hr { border-color: #1a2540 !important; }

/* ── section header ── */
.section-hdr {
    font-size: 11px;
    font-weight: 700;
    color: #2a4060;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 18px 0 10px;
    font-family: 'DM Mono', monospace;
}

/* ── denoise badge ── */
.dnoise-on  { color: #68d391; font-size: 12px; font-family:'DM Mono',monospace; }
.dnoise-off { color: #4a6080; font-size: 12px; font-family:'DM Mono',monospace; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* sliders */
[data-testid="stSlider"] label { color: #4a7090; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌙 Zero-DCE++")
    st.markdown('<div style="color:#2a4060;font-size:12px;font-family:\'DM Mono\',monospace;">SE-BLOCK · CHANNEL ATTENTION</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Model Version Selector ──────────────────────
    st.markdown('<div class="section-hdr">Model Version</div>', unsafe_allow_html=True)

    version_names = list(MODEL_REGISTRY.keys())
    selected_version = st.radio(
        label="Select version",
        options=version_names,
        index=0,
        label_visibility="collapsed",
    )

    # Show card for selected version
    ver = MODEL_REGISTRY[selected_version]
    st.markdown(
        f'<div class="ver-card selected">'
        f'<div class="ver-tag" style="background:{ver["tag_color"]};color:{ver["tag_text"]}">{ver["tag"]}</div>'
        f'<div class="ver-name">{ver["file"]}</div>'
        f'<div class="ver-desc">{ver["desc"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Check if .pth exists
    pth_path = ver["file"]
    model_ready = os.path.exists(pth_path)
    if not model_ready:
        st.warning(f"⚠️ `{pth_path}` not found.\nPlace the `.pth` file in the same folder as `app.py`.")
    else:
        st.success(f"✅ Checkpoint found")

    st.markdown("---")

    # ── Enhancement Settings ────────────────────────
    st.markdown('<div class="section-hdr">Enhancement Settings</div>', unsafe_allow_html=True)

    iterations = st.slider(
        "Curve Iterations",
        min_value=1, max_value=16, value=8,
        help="How many times LE(x;α)=x+α·x·(1−x) is applied. More = brighter output.",
    )

    max_size = st.select_slider(
        "Max Image Size (px)",
        options=[256, 512, 768, 1024, 1280, 1600, 2048],
        value=1024,
        help="Images larger than this are downscaled before inference.",
    )

    st.markdown("---")

    # ── Denoising Settings ──────────────────────────
    st.markdown('<div class="section-hdr">Denoising (Bilateral Filter)</div>', unsafe_allow_html=True)

    denoise_on = st.toggle(
        "Apply bilateral denoising",
        value=True,
        help="Edge-preserving noise reduction applied AFTER enhancement. "
             "Recommended for real-world photos. Turn off for LOL-v2 metric evaluation.",
    )

    if denoise_on:
        st.markdown(
            '<span class="dnoise-on">● Denoising ON</span> — '
            '<span style="color:#2a4060;font-size:11px;">bilateral filter active</span>',
            unsafe_allow_html=True,
        )
        with st.expander("Filter parameters", expanded=False):
            d_val     = st.slider("Diameter (d)",         5, 15,  9, 2,
                                  help="Filter neighbourhood size. Larger = smoother but slower.")
            sig_color = st.slider("Sigma Color",          25, 150, 75, 5,
                                  help="How similar colours must be to be blended. Higher = more smoothing.")
            sig_space = st.slider("Sigma Space",          25, 150, 75, 5,
                                  help="Spatial influence radius. Higher = wider smooth area.")
    else:
        st.markdown(
            '<span class="dnoise-off">○ Denoising OFF</span> — '
            '<span style="color:#2a4060;font-size:11px;">raw curve output</span>',
            unsafe_allow_html=True,
        )
        d_val = sig_color = sig_space = None

    st.markdown("---")

    # ── Architecture Info ───────────────────────────
    st.markdown('<div class="section-hdr">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="color:#2a4060;font-size:12px;line-height:2;font-family:'DM Mono',monospace;">
backbone &nbsp;→ Zero-DCE++ (7 conv)<br>
novelty &nbsp;&nbsp;→ SE-Block after Conv3<br>
output &nbsp;&nbsp;&nbsp;→ 24 α maps (8×RGB)<br>
loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ spa + exp + col + tv<br>
denoise &nbsp;&nbsp;→ bilateral (inference)
</div>
""", unsafe_allow_html=True)

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f'<div style="color:#2a4060;font-size:11px;margin-top:12px;font-family:\'DM Mono\',monospace;">device → {device_str}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_ready:
    try:
        model, device = load_model(pth_path)
    except Exception as e:
        st.error(f"❌ Failed to load `{pth_path}`: {e}")
        model = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Title
st.markdown("""
<div style="margin-bottom:8px;">
  <span style="font-size:32px;font-weight:800;color:#c8d8f0;letter-spacing:-0.5px;">
    Low-Light Image Enhancer
  </span>
  <span style="font-size:14px;color:#2a4060;margin-left:12px;font-family:'DM Mono',monospace;">
    Zero-DCE++ · SE-Block
  </span>
</div>
<div style="color:#2a4060;font-size:13px;margin-bottom:20px;">
  Select a model version in the sidebar, upload your dark images, and enhance.
</div>
""", unsafe_allow_html=True)

# Status bar
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    model_lbl = selected_version.split("—")[0].strip() if "—" in selected_version else selected_version
    st.markdown(f'<div class="stat-box"><div class="stat-val">{model_lbl}</div><div class="stat-lbl">Active Model</div></div>', unsafe_allow_html=True)
with col_s2:
    status_str = "Ready" if (model is not None) else "Not Loaded"
    status_col = "#68d391" if model else "#f6ad55"
    st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{status_col}">{status_str}</div><div class="stat-lbl">Model Status</div></div>', unsafe_allow_html=True)
with col_s3:
    dn_str = f"ON (d={d_val}, σ={sig_color})" if denoise_on else "OFF"
    st.markdown(f'<div class="stat-box"><div class="stat-val" style="font-size:14px">{dn_str}</div><div class="stat-lbl">Bilateral Denoise</div></div>', unsafe_allow_html=True)

st.markdown("---")

if not model_ready:
    st.info(f"👈 Place `{ver['file']}` in the same folder as `app.py`, then refresh.")
elif model is None:
    st.error("Model failed to load. Check the error above.")
else:
    # ── Image upload ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Upload Images</div>', unsafe_allow_html=True)
    uploaded_images = st.file_uploader(
        "Drop low-light images here",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_images:
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            run = st.button("✨  Enhance All")

        st.markdown("---")

        for idx, uploaded in enumerate(uploaded_images):
            img_orig = Image.open(uploaded).convert("RGB")
            W, H = img_orig.size

            # Downscale if needed
            if max(W, H) > max_size:
                scale = max_size / max(W, H)
                img_input = img_orig.resize(
                    (int(W * scale), int(H * scale)), Image.LANCZOS)
            else:
                img_input = img_orig

            st.markdown(f'<div style="font-size:15px;font-weight:700;color:#c8d8f0;margin-bottom:12px;">📷 {uploaded.name}</div>', unsafe_allow_html=True)

            # Stats row
            s1, s2, s3, s4, s5 = st.columns(5)
            orig_bright = avg_brightness(img_input)
            with s1:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{W}×{H}</div><div class="stat-lbl">Original</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{img_input.size[0]}×{img_input.size[1]}</div><div class="stat-lbl">Inference</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{orig_bright:.0f}</div><div class="stat-lbl">Input Brightness</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{iterations}</div><div class="stat-lbl">Curve Iters</div></div>', unsafe_allow_html=True)
            with s5:
                dn_label = f"d={d_val}" if denoise_on else "Off"
                st.markdown(f'<div class="stat-box"><div class="stat-val">{dn_label}</div><div class="stat-lbl">Denoise</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

            col_orig, col_enh = st.columns(2)

            with col_orig:
                st.markdown('<div class="img-label">📷 Original (low-light)</div>', unsafe_allow_html=True)
                st.image(img_input, use_container_width=True)

            with col_enh:
                enh_label = "✨ Enhanced" + (" + Bilateral Denoise" if denoise_on else "")
                st.markdown(f'<div class="img-label">{enh_label}</div>', unsafe_allow_html=True)

                cache_key = f"enh_{idx}_{selected_version}_{iterations}_{denoise_on}_{d_val}_{sig_color}_{sig_space}"

                if run or cache_key in st.session_state:
                    if run or cache_key not in st.session_state:
                        with st.spinner("Enhancing…"):
                            t0 = time.time()
                            result = run_enhance(
                                model, device, img_input,
                                iterations=iterations,
                                denoise=denoise_on,
                                d=d_val if denoise_on else 9,
                                sigma_color=sig_color if denoise_on else 75,
                                sigma_space=sig_space if denoise_on else 75,
                            )
                            elapsed = time.time() - t0
                        st.session_state[cache_key]               = result
                        st.session_state[cache_key + "_time"]     = elapsed

                    result  = st.session_state[cache_key]
                    elapsed = st.session_state.get(cache_key + "_time", 0)

                    st.image(result, use_container_width=True)

                    # Metrics
                    enh_bright = avg_brightness(result)
                    gain       = enh_bright - orig_bright

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Brightness Gain", f"+{gain:.1f}", delta=f"+{gain:.1f}")
                    with m2:
                        st.metric("Inference Time", f"{elapsed:.2f}s")
                    with m3:
                        st.metric("Output Brightness", f"{enh_bright:.1f}")

                    # Download
                    fname = uploaded.name.rsplit(".", 1)[0] + "_enhanced.png"
                    st.download_button(
                        label="⬇️  Download Enhanced PNG",
                        data=pil_to_bytes(result),
                        file_name=fname,
                        mime="image/png",
                        key=f"dl_{idx}_{selected_version}",
                    )
                else:
                    st.markdown(
                        '<div style="height:220px;display:flex;align-items:center;'
                        'justify-content:center;color:#1e3050;font-size:13px;'
                        'border:1px dashed #1a2540;border-radius:8px;'
                        'font-family:\'DM Mono\',monospace;">'
                        'click Enhance All to process →</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

    else:
        # Empty state
        st.markdown("""
<div style="text-align:center;padding:80px 0;color:#1a2540;">
  <div style="font-size:52px;margin-bottom:16px;">🌑</div>
  <div style="font-size:18px;font-weight:700;color:#2a4060;margin-bottom:8px;">
    No images uploaded yet
  </div>
  <div style="font-size:13px;color:#1a2540;font-family:'DM Mono',monospace;">
    drop your low-light photos above to get started
  </div>
</div>
""", unsafe_allow_html=True)