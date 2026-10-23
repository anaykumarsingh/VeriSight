
from pathlib import Path
import html
import math

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

# ============================================================
# VERISIGHT | AI IMAGE FORENSICS
# ============================================================

st.set_page_config(
    page_title="VeriSight | AI Image Forensics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_dedup.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BG = "#F7F3F0"
PAPER = "#FFFCF9"
SIDEBAR = "#F1E9E4"
MOCHA = "#6F4E37"
DARK = "#4B3427"
INK = "#39291F"
MUTED = "#83766E"
BORDER = "#E5DCD6"
LAVENDER = "#E8DFF5"
PURPLE = "#9475B5"
GREEN = "#438B72"
RED = "#A45E61"


def esc(value):
    return html.escape(str(value))


# ============================================================
# SVG ICONS
# ============================================================

def icon(name, size=22):
    paths = {
        "search": """
            <circle cx="10.8" cy="10.8" r="6.8"/>
            <path d="m16 16 5 5"/>
        """,
        "home": """
            <path d="m3 10 9-7 9 7"/>
            <path d="M5.5 9v11h13V9"/>
            <path d="M9.5 20v-6h5v6"/>
        """,
        "book": """
            <path d="M4 5a3 3 0 0 1 3-3h5v19H7a3 3 0 0 0-3 2z"/>
            <path d="M20 5a3 3 0 0 0-3-3h-5v19h5a3 3 0 0 1 3 2z"/>
        """,
        "info": """
            <circle cx="12" cy="12" r="9"/>
            <path d="M12 11v5"/>
            <path d="M12 7.5h.01"/>
        """,
        "image": """
            <rect x="3" y="3" width="18" height="18" rx="3"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <path d="m21 15-5-5L5 21"/>
        """,
        "cpu": """
            <rect x="6" y="6" width="12" height="12" rx="2"/>
            <path d="M9 1v3M15 1v3M9 20v3M15 20v3"/>
            <path d="M1 9h3M1 15h3M20 9h3M20 15h3"/>
        """,
        "shield": """
            <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/>
            <path d="m9 12 2 2 4-4"/>
        """,
        "check": '<path d="m5 12 4 4L19 6"/>',
        "alert": """
            <path d="m12 3 10 18H2L12 3z"/>
            <path d="M12 9v5M12 17h.01"/>
        """,
        "sparkle": """
            <path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3z"/>
            <path d="m19 14 1.2 2.8L23 18l-2.8 1.2L19 22l-1.2-2.8L15 18l2.8-1.2L19 14z"/>
        """,
        "target": """
            <circle cx="12" cy="12" r="9"/>
            <circle cx="12" cy="12" r="5"/>
            <circle cx="12" cy="12" r="1"/>
        """,
        "file": """
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
            <path d="M13 2v7h7"/>
        """,
        "upload": """
            <path d="M12 16V4"/>
            <path d="m7 9 5-5 5 5"/>
            <path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/>
        """,
    }

    return (
        f'<svg class="vs-icon" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">'
        f'{paths.get(name, paths["info"])}</svg>'
    )


# ============================================================
# CSS AND TYPOGRAPHY
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {BG};
        --paper: {PAPER};
        --mocha: {MOCHA};
        --dark: {DARK};
        --ink: {INK};
        --muted: {MUTED};
        --border: {BORDER};
        --lavender: {LAVENDER};
        --purple: {PURPLE};
        --green: {GREEN};
        --red: {RED};
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }}

    .stApp {{
        background: var(--bg);
    }}

    h1, h2, h3 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--ink);
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1550px;
        padding: 2rem 2rem 1rem;
    }}

    .vs-icon {{
        display: inline-block;
        vertical-align: middle;
        flex-shrink: 0;
    }}

    /* SIDEBAR */

    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid var(--border);
        min-width: 245px;
    }}

    section[data-testid="stSidebar"] > div {{
        padding: 1.4rem 1rem;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 5px 4px 22px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }}

    .brand-icon {{
        background: var(--mocha);
        color: white;
        width: 45px;
        height: 45px;
        min-width: 45px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .brand-name {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 25px;
        color: var(--ink);
        line-height: 1.1;
    }}

    .brand-sub {{
        font-size: 9px;
        letter-spacing: 1.6px;
        color: var(--muted);
        margin-top: 5px;
        white-space: nowrap;
    }}

    .sidebar-section {{
        font-size: 10px;
        letter-spacing: 2px;
        color: var(--muted);
        padding: 0 10px 10px;
    }}

    section[data-testid="stSidebar"] div.stButton > button {{
        width: 100%;
        min-height: 45px;
        border-radius: 11px;
        text-align: left;
        justify-content: flex-start;
        font-size: 13px;
        margin: 4px 0;
        box-shadow: none;
    }}

    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
        background: #E8DFF0;
        color: var(--dark);
        border: 1px solid #DCCBE9;
        font-weight: 700;
    }}

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {{
        background: transparent;
        color: var(--ink);
        border: 1px solid transparent;
    }}

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {{
        background: #EDE3F4;
        border-color: #E1D2EB;
    }}

    .sidebar-quote {{
        font-family: 'Playfair Display', Georgia, serif;
        font-style: italic;
        font-size: 22px;
        line-height: 1.3;
        margin-top: 32px;
        color: var(--mocha);
    }}

    .sidebar-note {{
        font-size: 12px;
        line-height: 1.8;
        color: var(--muted);
        margin-top: 13px;
    }}

    /* HERO */

    .eyebrow {{
        font-size: 10px;
        letter-spacing: 2.8px;
        color: var(--mocha);
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .hero-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(39px, 4.2vw, 64px);
        line-height: 1.05;
        letter-spacing: -1.5px;
        margin: 0 0 16px;
        color: var(--ink);
    }}

    .hero-title span {{
        color: var(--purple);
    }}

    .hero-description {{
        font-size: 13px;
        line-height: 1.9;
        color: var(--muted);
        max-width: 600px;
    }}

    .hero-art {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, #E8DFF5, #F1E6DF 58%, #D8C9BB);
        border: 1px solid #E6DAD5;
        border-radius: 23px;
        min-height: 230px;
        padding: 25px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .hero-art:before {{
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        border: 1px solid rgba(111,78,55,.25);
        border-radius: 50%;
        right: 25px;
        top: 26px;
        box-shadow: 0 0 0 17px rgba(255,252,249,.14),
                    0 0 0 36px rgba(255,252,249,.12);
    }}

    .hero-art:after {{
        content: "";
        position: absolute;
        width: 76px;
        height: 76px;
        border: 1px solid rgba(111,78,55,.4);
        border-radius: 50%;
        right: 72px;
        top: 73px;
        background: rgba(255,252,249,.23);
    }}

    .hero-art-small, .hero-art-title, .hero-art-foot {{
        position: relative;
        z-index: 1;
    }}

    .hero-art-small {{
        font-size: 9px;
        letter-spacing: 2.5px;
        color: var(--mocha);
    }}

    .hero-art-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 39px;
        line-height: 1.1;
        color: var(--mocha);
    }}

    .hero-art-foot {{
        font-size: 12px;
        color: var(--mocha);
    }}

    /* CARDS AND PANELS */

    .stat-card {{
        background: var(--paper);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px;
        min-height: 84px;
    }}

    .stat-label {{
        font-size: 9px;
        letter-spacing: 1.6px;
        color: var(--muted);
        text-transform: uppercase;
    }}

    .stat-value {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 19px;
        color: var(--ink);
        margin-top: 8px;
        overflow-wrap: anywhere;
    }}

    .panel {{
        background: var(--paper);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 21px;
        box-shadow: 0 4px 14px rgba(75,52,39,.035);
        margin-bottom: 15px;
    }}

    .panel-heading {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 24px;
        color: var(--ink);
    }}

    .panel-subtitle {{
        font-size: 12px;
        color: var(--muted);
        margin-top: 6px;
        margin-bottom: 15px;
        line-height: 1.8;
    }}

    .panel-icon-title {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .icon-disc {{
        width: 38px;
        height: 38px;
        min-width: 38px;
        border-radius: 50%;
        background: #F0E8E2;
        color: var(--mocha);
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    /* RESULT */

    .result-real, .result-fake {{
        border-radius: 14px;
        padding: 20px;
    }}

    .result-real {{
        background: #EDF4EF;
        border: 1px solid #D6E4D9;
    }}

    .result-fake {{
        background: #F8EEEE;
        border: 1px solid #EBD9D9;
    }}

    .verdict {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 36px;
        line-height: 1.15;
        margin: 7px 0;
    }}

    .result-caption {{
        font-size: 12px;
        line-height: 1.8;
        color: var(--muted);
    }}

    .prob-label {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        font-size: 12px;
        margin: 16px 0 7px;
    }}

    .progress-track {{
        background: #EDE8E4;
        height: 8px;
        border-radius: 20px;
        overflow: hidden;
    }}

    .progress-fill {{
        height: 100%;
        border-radius: 20px;
    }}

    .empty-report {{
        background: #F3EDF7;
        border: 1px solid #E2D7EC;
        border-radius: 14px;
        padding: 28px 18px;
        text-align: center;
    }}

    /* STREAMLIT CONTROLS */

    div.stButton > button[kind="primary"] {{
        background: linear-gradient(100deg, #654633, #98745D);
        color: white;
        border: none;
        border-radius: 11px;
        min-height: 46px;
        font-size: 14px;
        font-weight: 600;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: #4B3427;
        border: none;
    }}

    div[data-testid="stFileUploader"] {{
        background: #FBF8FC;
        border: 1px dashed #CDB9DF;
        border-radius: 12px;
        padding: 8px;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--paper);
    }}

    .disclaimer {{
        border-top: 1px solid var(--border);
        margin-top: 25px;
        padding-top: 13px;
        font-size: 10px;
        line-height: 1.8;
        color: var(--muted);
    }}

    @media (max-width: 760px) {{
        .block-container {{ padding: 1rem .8rem; }}
        .hero-title {{ font-size: 41px; }}
        .panel {{ padding: 15px; }}
        .hero-art {{ min-height: 170px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    class_names = checkpoint.get(
        "class_names", ["Real", "Fake"]
    )
    image_size = int(checkpoint.get("image_size", 224))

    return model, class_names, image_size


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def analyze_image(image, model, class_names, image_size):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = get_transform(image_size)(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    scores = {
        str(class_names[i]).strip().lower(): float(probabilities[i])
        for i in range(len(class_names))
    }

    if "real" not in scores or "fake" not in scores:
        raise ValueError(
            f"Expected Real/Fake classes, got {class_names}"
        )

    real_score = scores["real"]
    fake_score = scores["fake"]
    prediction = "Real" if real_score >= fake_score else "Fake"
    confidence = max(real_score, fake_score)

    return prediction, confidence, real_score, fake_score


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-icon">{icon("search", 25)}</div>
            <div>
                <div class="brand-name">VeriSight</div>
                <div class="brand-sub">AI IMAGE FORENSICS</div>
            </div>
        </div>
        <div class="sidebar-section">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )

    navigation = [
        ("Home", "home", "⌂"),
        ("How it works", "book", "▤"),
        ("About", "info", "ⓘ"),
    ]

    for label, icon_name, glyph in navigation:
        if st.button(
            f"{glyph}   {label}",
            key=f"nav_{label}",
            type=(
                "primary"
                if st.session_state.page == label
                else "secondary"
            ),
            width="stretch",
        ):
            st.session_state.page = label
            st.rerun()

    st.markdown(
        """
        <div style="height:1px;background:#E0D5CE;margin:25px 8px"></div>
        <div class="sidebar-quote">Truth lives<br>in the details.</div>
        <div class="sidebar-note">
            A research prototype for exploring image authenticity
            with deep learning.
        </div>
        <div style="margin-top:26px;color:#A58BC1;font-size:20px">✧</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = None
class_names = ["Real", "Fake"]
image_size = 224
model_error = None

try:
    model, class_names, image_size = load_model()
except Exception as exc:
    model_error = str(exc)


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    hero_left, hero_right = st.columns([1.35, 1], gap="large")

    with hero_left:
        st.markdown(
            """
            <div class="eyebrow">AI-powered image analysis</div>
            <div class="hero-title">
                See beyond<br>the <span>image.</span>
            </div>
            <div class="hero-description">
                Upload an image and let our deep learning model analyze
                it for signs of manipulation. Explore the prediction,
                class scores and technical details in one place.
            </div>
            <div style="display:flex;align-items:center;gap:17px;
            flex-wrap:wrap;margin-top:20px;font-size:11px;color:#6F4E37">
                <span>✧ &nbsp;Fast analysis</span>
                <span>◎ &nbsp;Deep learning</span>
                <span>♢ &nbsp;Real vs Fake</span>
                <span>▦ &nbsp;EfficientNet-B0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            """
            <div class="hero-art">
                <div class="hero-art-small">LOOK CLOSER · THINK DEEPER</div>
                <div class="hero-art-title">Real or<br>fake?</div>
                <div class="hero-art-foot">Every pixel tells a story.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if model_error:
        st.error(
            "The model could not be loaded. Check the checkpoint "
            "and model architecture."
        )
        with st.expander("Technical error"):
            st.code(model_error)

    status = "Ready" if model is not None else "Unavailable"

    stats = [
        ("MODEL", "EfficientNet-B0"),
        ("INPUT SIZE", f"{image_size} × {image_size}"),
        ("DEVICE", str(DEVICE).upper()),
        ("STATUS", status),
    ]

    stat_cols = st.columns(4, gap="small")

    for col, (label, value) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{esc(label)}</div>
                    <div class="stat-value">{esc(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    input_col, report_col = st.columns([1, 1.15], gap="medium")

    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    with input_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-icon-title">
                    <div class="icon-disc">{icon("image", 20)}</div>
                    <div class="panel-heading">01 / Image Input</div>
                </div>
                <div class="panel-subtitle">
                    Upload a JPG, PNG or WebP image for inspection.
                </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="Supported formats: JPG, JPEG, PNG and WebP.",
            key="image_upload",
        )

        image = None

        if uploaded is not None:
            try:
                image = ImageOps.exif_transpose(
                    Image.open(uploaded)
                ).convert("RGB")

                st.image(
                    image,
                    caption=uploaded.name,
                    width="stretch",
                )

                width, height = image.size
                size_kb = len(uploaded.getvalue()) / 1024

                st.markdown(
                    f"""
                    <div class="stat-card" style="margin-top:12px">
                        <div class="panel-icon-title">
                            <div class="icon-disc">{icon("file", 18)}</div>
                            <div class="stat-label">IMAGE INFORMATION</div>
                        </div>
                        <div style="font-size:12px;line-height:2.1;margin-top:10px">
                            <b>Filename:</b> {esc(uploaded.name)}<br>
                            <b>Dimensions:</b> {width} × {height} px<br>
                            <b>File size:</b> {size_kb:.1f} KB
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception:
                st.error(
                    "This image could not be opened. Please choose another."
                )
                image = None

        st.markdown("</div>", unsafe_allow_html=True)

        analyze_clicked = st.button(
            "✧  Analyze Image",
            type="primary",
            width="stretch",
            disabled=(image is None or model is None),
            key="analyze_image",
        )

        if st.button(
            "Clear result",
            width="stretch",
            key="clear_result",
        ):
            st.session_state.analysis = None
            st.rerun()

        if analyze_clicked and image is not None and model is not None:
            try:
                with st.spinner("Analyzing your image..."):
                    prediction, confidence, real, fake = analyze_image(
                        image, model, class_names, image_size
                    )

                st.session_state.analysis = {
                    "filename": uploaded.name,
                    "prediction": prediction,
                    "confidence": confidence,
                    "real": real,
                    "fake": fake,
                    "dimensions": image.size,
                }

                st.rerun()

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    # --------------------------------------------------------
    # FORENSIC REPORT
    # --------------------------------------------------------

    with report_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-icon-title">
                    <div class="icon-disc">{icon("shield", 20)}</div>
                    <div class="panel-heading">02 / Forensic Report</div>
                </div>
                <div class="panel-subtitle">
                    Prediction and class-score breakdown.
                </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state.analysis

        if result is None:
            st.markdown(
                f"""
                <div class="empty-report">
                    <div style="color:{PURPLE};display:flex;
                    justify-content:center;margin-bottom:12px">
                        {icon("sparkle", 30)}
                    </div>
                    <div style="font-family:'Playfair Display',Georgia,serif;
                    font-size:24px;color:{DARK};margin-bottom:10px">
                        Your report awaits
                    </div>
                    <div style="font-size:12px;color:{MUTED};line-height:1.9">
                        Upload an image and select Analyze Image to view
                        the model's prediction and class scores here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            prediction = result["prediction"]
            confidence = result["confidence"]
            real_score = result["real"]
            fake_score = result["fake"]

            is_real = prediction == "Real"
            result_class = "result-real" if is_real else "result-fake"
            result_color = GREEN if is_real else RED
            ring_color = result_color
            result_icon = "check" if is_real else "alert"

            circumference = 2 * math.pi * 43
            dash = circumference * confidence

            st.markdown(
                f"""
                <div class="{result_class}">
                    <div style="display:flex;gap:14px;align-items:center">
                        <div style="width:100px;height:100px;min-width:100px;
                        position:relative;display:flex;align-items:center;
                        justify-content:center">
                            <svg width="100" height="100" viewBox="0 0 100 100"
                            style="position:absolute;inset:0;transform:rotate(-90deg)">
                                <circle cx="50" cy="50" r="43"
                                stroke="#E3DDD7" stroke-width="7" fill="none"/>
                                <circle cx="50" cy="50" r="43"
                                stroke="{ring_color}" stroke-width="7"
                                stroke-linecap="round" fill="none"
                                stroke-dasharray="{dash:.2f} {circumference:.2f}"/>
                            </svg>
                            <div style="position:relative;text-align:center">
                                <div style="font-family:'Playfair Display',Georgia,serif;
                                font-size:21px;color:{result_color}">
                                    {confidence*100:.1f}%
                                </div>
                                <div style="font-size:8px;letter-spacing:1px;color:{MUTED}">
                                    SCORE
                                </div>
                            </div>
                        </div>
                        <div style="flex:1;min-width:0">
                            <div style="font-size:9px;letter-spacing:2px;color:{MUTED}">
                                MODEL PREDICTION
                            </div>
                            <div class="verdict" style="color:{result_color}">
                                {esc(prediction).upper()}
                            </div>
                            <div style="font-size:11px;color:{MUTED}">
                                {icon(result_icon, 14)} &nbsp;Selected-class score
                            </div>
                        </div>
                    </div>
                    <div class="result-caption" style="margin-top:12px">
                        The model classified this image as
                        <b>{esc(prediction.lower())}</b>.<br>
                        This is a prediction, not a verified authenticity finding.
                    </div>
                </div>
                <div class="panel-heading" style="font-size:20px;margin-top:22px">
                    Class Probabilities
                </div>
                """,
                unsafe_allow_html=True,
            )

            for label, score, color in [
                ("REAL", real_score, GREEN),
                ("FAKE", fake_score, PURPLE),
            ]:
                st.markdown(
                    f"""
                    <div class="prob-label">
                        <span style="display:flex;align-items:center;gap:8px">
                            <span style="color:{color}">
                                {icon("shield" if label == "REAL" else "alert", 16)}
                            </span>
                            {label}
                        </span>
                        <span>{score*100:.2f}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill"
                        style="width:{score*100:.2f}%;background:{color}">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            score_col, dimension_col = st.columns(2)

            with score_col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">SELECTED-CLASS SCORE</div>
                        <div class="stat-value">{confidence*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with dimension_col:
                width, height = result["dimensions"]
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">ORIGINAL DIMENSIONS</div>
                        <div class="stat-value">{width} × {height}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Technical details"):
                st.write("**Model:** EfficientNet-B0")
                st.write(f"**Checkpoint:** `{MODEL_PATH.name}`")
                st.write(f"**Input size:** {image_size} × {image_size}")
                st.write(f"**Device:** {DEVICE}")
                st.write(f"**Filename:** {result['filename']}")
                st.write(
                    "**Score note:** Softmax scores are not calibrated "
                    "probabilities of authenticity."
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="disclaimer">
            {icon("shield", 15)} &nbsp; RESEARCH PROTOTYPE —
            This model can make mistakes, especially on images unlike its
            training data. Blur, compression and image content may affect
            results. Do not use the prediction alone to establish authenticity
            or make consequential decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOW IT WORKS
# ============================================================

elif st.session_state.page == "How it works":

    st.markdown(
        """
        <div class="eyebrow">Behind the analysis</div>
        <div class="hero-title">How it <span>works.</span></div>
        <div class="hero-description">
            A simple overview of VeriSight's image-analysis pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Upload", "Choose a supported image from your device.", "upload"),
        ("02", "Preprocess", "The image is resized and normalized.", "image"),
        ("03", "Analyze", "EfficientNet-B0 produces Real and Fake class scores.", "cpu"),
        ("04", "Review", "Read the prediction and compare both scores.", "shield"),
    ]

    for number, title, description, icon_name in steps:
        st.markdown(
            f"""
            <div class="stat-card" style="margin:12px 0">
                <div style="display:flex;align-items:center;gap:17px">
                    <div class="icon-disc">{icon(icon_name, 21)}</div>
                    <div style="font-family:'Playfair Display',Georgia,serif;
                    font-size:25px;color:{PURPLE}">{number}</div>
                    <div style="flex:1">
                        <div class="panel-heading" style="font-size:21px">
                            {esc(title)}
                        </div>
                        <div style="font-size:12px;line-height:1.8;
                        color:{MUTED};margin-top:5px">
                            {esc(description)}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "The model does not prove whether an image is genuine. "
        "Predictions can be wrong, including for ordinary photographs."
    )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About":

    st.markdown(
        """
        <div class="eyebrow">Our project</div>
        <div class="hero-title">About <span>VeriSight.</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stat-card" style="padding:25px">
            <div class="panel-icon-title">
                <div class="icon-disc">{icon("search", 21)}</div>
                <div class="panel-heading">AI Image Forensics</div>
            </div>
            <p style="font-size:13px;line-height:2;color:{MUTED};margin-top:16px">
                VeriSight is a research prototype that explores binary
                real-versus-fake image classification using deep learning.
                It uses an EfficientNet-B0 model to generate a prediction
                and class scores for an uploaded image.
            </p>
            <div class="panel-heading" style="font-size:21px;margin-top:22px">
                Important limitations
            </div>
            <p style="font-size:13px;line-height:2;color:{MUTED}">
                Results depend on the training data and image characteristics.
                Unfamiliar image sources, blur, compression and other changes
                can affect predictions. The displayed score is not a guarantee
                of correctness or a calibrated probability that an image is
                authentic.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("**Technology stack**")
    st.markdown(
        "`Python` · `Streamlit` · `PyTorch` · `Torchvision` · `Pillow`"
    )
