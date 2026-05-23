import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import time

# ─────────────────────────────────────────────
# Model architecture (must match training code)
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DCENet(nn.Module):
    """Zero-DCE++ with Channel Attention after Conv3"""
    def __init__(self, use_attention=True):
        super(DCENet, self).__init__()
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
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
        x5 = torch.cat([x4, x3], 1)
        x5 = self.relu(self.conv5(x5))
        x6 = torch.cat([x5, x2], 1)
        x6 = self.relu(self.conv6(x6))
        x7 = torch.cat([x6, x1], 1)
        alpha = self.conv7(x7)
        return torch.tanh(alpha)


def enhance_image_with_curves(x, alpha, iterations=8):
    """Apply Zero-DCE curve: LE(x; α) = x + α·x·(1−x), N times"""
    enhanced = x.clone()
    for i in range(iterations):
        alpha_i = alpha[:, i * 3:(i + 1) * 3, :, :]
        enhanced = enhanced + alpha_i * enhanced * (1 - enhanced)
    return torch.clamp(enhanced, 0, 1)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

@st.cache_resource
def load_model(weights_path: str, device: torch.device):
    model = DCENet(use_attention=True).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def enhance(model, img: Image.Image, device: torch.device, iterations: int = 8) -> Image.Image:
    tensor = pil_to_tensor(img, device)
    with torch.no_grad():
        alpha = model(tensor)
        enhanced = enhance_image_with_curves(tensor, alpha, iterations=iterations)
    return tensor_to_pil(enhanced)


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Zero-DCE++ Low-Light Enhancer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; }

    /* Cards */
    .img-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3148;
        text-align: center;
    }
    .img-card h4 {
        color: #a0aec0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 3px;
    }
    .badge-blue  { background:#1e3a5f; color:#90cdf4; }
    .badge-green { background:#1a3a2a; color:#68d391; }
    .badge-purple{ background:#2d1b69; color:#b794f4; }

    /* Stat box */
    .stat-box {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-val { font-size: 22px; font-weight: 700; color: #e2e8f0; }
    .stat-lbl { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.8px; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3d4263 !important;
        border-radius: 12px !important;
        background: #1a1d27 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] { padding-top: 4px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌙 Zero-DCE++ SE-Block")
    st.markdown(
        '<span class="badge badge-blue">Zero-DCE++</span>'
        '<span class="badge badge-purple">SE-Block</span>'
        '<span class="badge badge-green">Channel Attention</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### ⚙️ Model Weights")
    weights_file = st.file_uploader(
        "Upload `.pth` file",
        type=["pth"],
        help="Upload best_zerodce_seblock_v2.pth (or any compatible checkpoint)",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Enhancement Settings")

    iterations = st.slider(
        "Curve Iterations",
        min_value=1, max_value=16, value=8,
        help="Number of times the LE(x;α) curve is applied. More = brighter."
    )

    max_size = st.select_slider(
        "Max Image Size (px)",
        options=[256, 512, 768, 1024, 1280, 1600, 2048],
        value=1024,
        help="Images larger than this will be resized before inference."
    )

    st.markdown("---")
    st.markdown("### ℹ️ Architecture")
    st.markdown("""
- **Backbone**: Zero-DCE++ (7 conv layers)  
- **Novelty**: SE-Block after Conv3  
- **Output**: 24 α maps → 8 × RGB curves  
- **Loss**: L_spa + 10·L_exp + 5·L_col + 1600·L_tv  
    """)

    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Device:** `{device_name}`")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

st.markdown("# 🌙 Low-Light Image Enhancer")
st.markdown("Upload a model checkpoint, then enhance your dark images using **Zero-DCE++ with SE-Block channel attention**.")
st.markdown("---")

# ── Model loading state ──────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

if weights_file is not None:
    # Save uploaded weights to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(weights_file.read())
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path, device)
        st.success(f"✅ Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
    finally:
        os.unlink(tmp_path)
else:
    st.info("👈 Upload your `.pth` weights file in the sidebar to get started.")

st.markdown("---")

# ── Image upload ─────────────────────────────
st.markdown("### 📁 Upload Images")
uploaded_images = st.file_uploader(
    "Drop low-light images here (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_images:
    # ── Enhance button ───────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run = st.button("✨ Enhance All", disabled=(model is None))

    st.markdown("---")

    for idx, uploaded in enumerate(uploaded_images):
        img_orig = Image.open(uploaded).convert("RGB")
        W, H = img_orig.size

        # Resize if needed
        if max(W, H) > max_size:
            scale = max_size / max(W, H)
            img_input = img_orig.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        else:
            img_input = img_orig

        st.markdown(f"#### 🖼️ {uploaded.name}")

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{W}×{H}</div><div class="stat-lbl">Original Size</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{img_input.size[0]}×{img_input.size[1]}</div><div class="stat-lbl">Inference Size</div></div>', unsafe_allow_html=True)
        with s3:
            avg_brightness = int(np.array(img_input).mean())
            st.markdown(f'<div class="stat-box"><div class="stat-val">{avg_brightness}</div><div class="stat-lbl">Avg Brightness</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{iterations}</div><div class="stat-lbl">Curve Iters</div></div>', unsafe_allow_html=True)

        st.markdown("")

        col_orig, col_enh = st.columns(2)

        with col_orig:
            st.markdown('<div class="img-card"><h4>📷 Original (Low-Light)</h4></div>', unsafe_allow_html=True)
            st.image(img_input, use_container_width=True)

        with col_enh:
            st.markdown('<div class="img-card"><h4>✨ Enhanced</h4></div>', unsafe_allow_html=True)

            if model is not None and (run or f"enhanced_{idx}" in st.session_state):
                if run or f"enhanced_{idx}" not in st.session_state:
                    with st.spinner("Enhancing..."):
                        t0 = time.time()
                        enhanced_img = enhance(model, img_input, device, iterations=iterations)
                        elapsed = time.time() - t0
                    st.session_state[f"enhanced_{idx}"] = enhanced_img
                    st.session_state[f"elapsed_{idx}"] = elapsed

                enhanced_img = st.session_state[f"enhanced_{idx}"]
                st.image(enhanced_img, use_container_width=True)

                # Metrics
                orig_arr = np.array(img_input).astype(float)
                enh_arr  = np.array(enhanced_img).astype(float)
                brightness_gain = int(enh_arr.mean() - orig_arr.mean())
                elapsed = st.session_state.get(f"elapsed_{idx}", 0)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Brightness Gain", f"+{brightness_gain}", delta=f"+{brightness_gain}")
                with m2:
                    st.metric("Inference Time", f"{elapsed:.2f}s")
                with m3:
                    avg_enh = int(enh_arr.mean())
                    st.metric("Avg Brightness", f"{avg_enh}")

                # Download button
                dl_bytes = pil_to_bytes(enhanced_img, fmt="PNG")
                fname = uploaded.name.rsplit(".", 1)[0] + "_enhanced.png"
                st.download_button(
                    label="⬇️ Download Enhanced",
                    data=dl_bytes,
                    file_name=fname,
                    mime="image/png",
                    key=f"dl_{idx}",
                )
            else:
                st.markdown(
                    '<div style="height:200px;display:flex;align-items:center;'
                    'justify-content:center;color:#4a5568;font-size:14px;">'
                    '⬆️ Click "Enhance All" to process</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

elif model is not None:
    st.markdown(
        '<div style="text-align:center;padding:60px;color:#4a5568;">'
        '<div style="font-size:48px">📷</div>'
        '<div style="font-size:18px;margin-top:12px">Upload some low-light images to enhance</div>'
        '</div>',
        unsafe_allow_html=True
    )
