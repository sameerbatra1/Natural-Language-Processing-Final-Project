"""
app.py — Streamlit frontend for the Jigsaw Toxic Comment Classifier.

Run locally:
    streamlit run app.py

Deploy on HuggingFace Spaces (Streamlit SDK) without any code changes.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from config import (
    DEFAULT_THRESHOLD,
    LABEL_DISPLAY_NAMES,
    LABELS,
    MODEL_NAME,
    severity_color,
)
from inference import ToxicityPredictor

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Jigsaw Toxicity Classifier",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, clinical, minimal
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #e8e8e8;
}

/* Title block */
.title-block {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.title-block p {
    font-size: 0.9rem;
    color: #666;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Divider */
.thin-line {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

/* Text area override */
textarea {
    background: #111 !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.875rem !important;
}
textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
}

/* Analyze button */
div.stButton > button {
    background: #e8e8e8 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
div.stButton > button:hover {
    opacity: 0.85 !important;
    background: #ffffff !important;
}

/* Result card */
.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

/* Verdict banner */
.verdict-clean {
    background: #0d2b1a;
    border: 1px solid #1a4a2e;
    border-left: 4px solid #52b788;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #74c69d;
    font-size: 0.95rem;
}
.verdict-toxic {
    background: #2b0d0d;
    border: 1px solid #4a1a1a;
    border-left: 4px solid #e63946;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #e63946;
    font-size: 0.95rem;
}

/* Label row */
.label-row {
    display: flex;
    align-items: center;
    margin: 0.55rem 0;
    gap: 0.75rem;
}
.label-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #aaa;
    width: 160px;
    flex-shrink: 0;
}
.label-bar-bg {
    flex: 1;
    background: #1e1e1e;
    border-radius: 3px;
    height: 8px;
    position: relative;
    overflow: hidden;
}
.label-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.label-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}
.label-flag {
    font-size: 0.65rem;
    background: #e63946;
    color: white;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: 'IBM Plex Mono', monospace;
    flex-shrink: 0;
}

/* Example buttons row */
.example-label {
    font-size: 0.75rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111 !important;
}

/* Slider */
.stSlider [data-testid="stSliderThumb"] {
    background: #e8e8e8 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #333;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2rem 0 1rem 0;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model — cached so it loads only once across reruns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_predictor() -> ToxicityPredictor:
    return ToxicityPredictor(model_name=MODEL_NAME)

# ---------------------------------------------------------------------------
# Helper — render label bars
# ---------------------------------------------------------------------------

def render_label_bars(results: dict[str, float], threshold: float) -> None:
    html_rows = ""
    for label in LABELS:
        prob = results[label]
        pct  = prob * 100
        color = severity_color(prob)
        display = LABEL_DISPLAY_NAMES[label]
        flagged = prob >= threshold
        flag_html = '<span class="label-flag">FLAGGED</span>' if flagged else ""
        html_rows += f"""
        <div class="label-row">
            <span class="label-name">{display}</span>
            <div class="label-bar-bg">
                <div class="label-bar-fill"
                     style="width:{pct:.1f}%; background:{color};">
                </div>
            </div>
            <span class="label-pct">{pct:.1f}%</span>
            {flag_html}
        </div>"""
    st.markdown(html_rows, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper — Plotly radar chart
# ---------------------------------------------------------------------------

def render_radar(results: dict[str, float]) -> None:
    categories = [LABEL_DISPLAY_NAMES[l] for l in LABELS]
    values     = [results[l] for l in LABELS]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(230, 57, 70, 0.15)",
        line=dict(color="#e63946", width=1.5),
        marker=dict(color="#e63946", size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(color="#555", size=9, family="IBM Plex Mono"),
                gridcolor="#222",
                linecolor="#222",
            ),
            angularaxis=dict(
                tickfont=dict(color="#888", size=10, family="IBM Plex Mono"),
                gridcolor="#222",
                linecolor="#222",
            ),
        ),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        showlegend=False,
        margin=dict(t=30, b=30, l=40, r=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="title-block">
    <h1>🧪 JIGSAW TOXICITY CLASSIFIER</h1>
    <p>Multi-label NLP detection · 6 toxicity categories · BERT</p>
</div>
<hr class="thin-line">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Classification threshold",
        min_value=0.1, max_value=0.9,
        value=DEFAULT_THRESHOLD, step=0.05,
        help="Probability above which a label is flagged",
    )
    show_radar = st.checkbox("Show radar chart", value=True)
    st.markdown("---")
    st.markdown("""
    **Model:** `unitary/toxic-bert`  
    **Labels:** toxic · severe_toxic · obscene · threat · insult · identity_hate  
    **Backend:** PyTorch + HuggingFace Transformers  
    **Device:** CPU
    """)

# ---------------------------------------------------------------------------
# Example comments
# ---------------------------------------------------------------------------

EXAMPLES = [
    ("😇 Clean",      "Thank you for the helpful explanation, I really learned a lot!"),
    ("😠 Toxic",       "You are a complete idiot and should be banned from this site."),
    ("🔥 Severe",      "I will find you and make you regret ever posting that garbage."),
    ("🤔 Ambiguous",   "This is the worst article I have ever read. Absolutely terrible."),
]

st.markdown('<p class="example-label">Quick examples</p>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
selected_example = None
for col, (label, text) in zip(cols, EXAMPLES):
    if col.button(label, use_container_width=True):
        selected_example = text

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------

default_text = selected_example if selected_example else ""
comment = st.text_area(
    "Enter a comment to analyse",
    value=default_text,
    height=130,
    placeholder="Type or paste any comment here…",
    label_visibility="collapsed",
)

analyze_col, clear_col = st.columns([4, 1])
run = analyze_col.button("🔍 ANALYZE", use_container_width=True)
if clear_col.button("✕ Clear", use_container_width=True):
    comment = ""

# ---------------------------------------------------------------------------
# Inference + results
# ---------------------------------------------------------------------------

if run and comment.strip():
    with st.spinner("Loading model & running inference…"):
        predictor = load_predictor()
        results   = predictor.predict(comment)

    # --- Verdict banner ---
    flagged_labels = [l for l in LABELS if results[l] >= threshold]
    if flagged_labels:
        top_label   = max(results, key=lambda k: results[k])
        top_display = LABEL_DISPLAY_NAMES[top_label]
        top_pct     = results[top_label] * 100
        st.markdown(
            f'<div class="verdict-toxic">⚠ TOXIC CONTENT DETECTED &nbsp;·&nbsp; '
            f'Dominant: <strong>{top_display}</strong> ({top_pct:.1f}%)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict-clean">✓ No toxicity detected above threshold</div>',
            unsafe_allow_html=True,
        )

    # --- Label bars ---
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    render_label_bars(results, threshold)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Radar chart ---
    if show_radar:
        st.markdown('<hr class="thin-line">', unsafe_allow_html=True)
        render_radar(results)

    # --- Raw scores expander ---
    with st.expander("📋 Raw probability scores"):
        for label in LABELS:
            st.code(f"{label:<16}  {results[label]:.4f}", language="text")

elif run and not comment.strip():
    st.warning("Please enter some text before clicking Analyze.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<hr class="thin-line">
<div class="footer">
    JIGSAW TOXIC COMMENT CLASSIFICATION &nbsp;·&nbsp;
    NLP FINAL PROJECT &nbsp;·&nbsp;
    BERT-BASE-UNCASED &nbsp;·&nbsp; PyTorch + Streamlit
</div>
""", unsafe_allow_html=True)
