from pathlib import Path
import json
import re
import string
from datetime import datetime

import joblib
import pandas as pd
import altair as alt
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hate_speech_model.joblib"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
CONF_MATRIX_PATH = BASE_DIR / "models" / "confusion_matrix.png"

CLASS_COLORS = {
    # Fallback palette keyed by label text (edit to match your actual class_names.json values)
    "Hate Speech": "#e5484d",
    "Offensive Language": "#f5a524",
    "Neither": "#2fb344",
}
DEFAULT_COLOR = "#7c7fed"


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #e5e7eb;
            background: #0b1220;
        }

        .stApp, .main {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }

        .hero {
            padding: 2rem 2rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #5b75f0 0%, #8b5cf6 48%, #ec4899 100%);
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.18);
        }
        .hero h1 {
            color: white;
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0 0 0.5rem 0;
        }
        .hero p {
            color: rgba(255,255,255,0.92);
            font-size: 1rem;
            margin: 0;
            line-height: 1.6;
        }
        .hero .highlight {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
            color: rgba(255,255,255,0.9);
            font-size: 0.95rem;
        }

        .section-card {
            border-radius: 20px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .section-card h3 {
            margin-top: 0;
            color: #f8fafc;
        }

        .metric-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            text-align: left;
        }
        .metric-card .value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e0e7ff;
        }
        .metric-card .label {
            display: block;
            margin-top: 0.35rem;
            font-size: 0.85rem;
            color: rgba(226,232,240,0.75);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .result-card {
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.8rem;
            border-left: 6px solid var(--accent, #7c7fed);
            background: rgba(255,255,255,0.05);
        }
        .result-label {
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .result-sub {
            font-size: 0.95rem;
            color: rgba(226,232,240,0.8);
        }

        .sample-button button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.05);
            color: #f8fafc;
            padding: 0.85rem 0.9rem;
            font-weight: 600;
            transition: transform 0.2s ease, background 0.2s ease;
        }
        .sample-button button:hover {
            transform: translateY(-1px);
            background: rgba(255,255,255,0.12);
        }

        .history-item {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.04);
            margin-bottom: 0.65rem;
            font-size: 0.92rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .stButton>button {
            border-radius: 14px;
            font-weight: 700;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
        }
        .stButton>button:hover {
            opacity: 0.95;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Data / model loading
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = text.replace("#", "")
    text = re.sub(r"&(?:amp|lt|gt|quot);", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    model = joblib.load(MODEL_PATH)

    class_names = None
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = {int(k): v for k, v in json.load(f).items()}

    metrics = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, class_names, metrics


def predict_text(model, text: str, class_names: dict):
    """Predict class and confidence for a given input text."""
    text = clean_text(text)
    prediction = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]

    label = class_names.get(prediction, str(prediction)) if class_names else str(prediction)
    confidence = float(probabilities[prediction])

    probs = []
    for idx, prob in enumerate(probabilities):
        name = class_names.get(idx, str(idx)) if class_names else str(idx)
        probs.append({"Class": name, "Probability": float(prob)})

    probs.sort(key=lambda x: x["Probability"], reverse=True)
    return label, confidence, probs


def confidence_badge(confidence: float) -> tuple[str, str]:
    """Returns (message, color) for a confidence level."""
    if confidence >= 0.75:
        return "High confidence", "#2fb344"
    elif confidence >= 0.5:
        return "Moderate confidence", "#f5a524"
    return "Low confidence — treat with caution", "#e5484d"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def render_probability_chart(probs: list[dict]):
    df = pd.DataFrame(probs)
    color_values = [CLASS_COLORS.get(row['Class'], DEFAULT_COLOR) for row in probs]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Class:N", sort="-x", title=None),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(range=color_values),
                legend=None,
            ),
            tooltip=["Class", alt.Tooltip("Probability", format=".2%")],
        )
        .properties(height=150)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sample_buttons(samples: list[str]):
    cols = st.columns(3)
    for idx, sample in enumerate(samples):
        if cols[idx % 3].button(sample, key=f"sample_{idx}"):
            st.session_state.user_text = sample


def main():
    st.set_page_config(page_title="Hate Speech Detection System", page_icon="🛡️", layout="wide")
    inject_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    model, class_names, metrics = load_model()

    # ---- Hero header ----
    header_left, header_right = st.columns([3, 1], gap="large")
    with header_left:
        st.markdown(
            """
            <div class="hero">
                <h1>🛡️ Hate Speech Detection System</h1>
                <p>Analyze text for hate speech, abusive language, or neutral sentiment with an intuitive web interface.</p>
                <div class="highlight">Fast TF-IDF classification · confidence-driven results · ready for academic presentation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown("## Project Snapshot")
        st.markdown("""
            - Dataset: Davidson hate-speech corpus
            - Model: Logistic Regression
            - Visualization: Altair probability chart
            - Built for university presentation
        """)

    if model is None:
        st.warning("Model not found. Run `python train.py` first to create the trained model files.")
        st.stop()

    # ---- Sidebar: model info ----
    with st.sidebar:
        st.subheader("📊 Model Info")
        if metrics is not None:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{metrics.get('accuracy', 0) * 100:.1f}%</div>
                    <div class="label">Trained Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{metrics.get('train_size', 0)}</div>
                    <div class="label">Training examples</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{metrics.get('test_size', 0)}</div>
                    <div class="label">Validation examples</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if CONF_MATRIX_PATH.exists():
            st.markdown("---")
            st.caption("Confusion Matrix")
            st.image(str(CONF_MATRIX_PATH), use_container_width=True)

        st.markdown("---")
        st.caption(f"Session predictions: {len(st.session_state.history)}")
        if st.session_state.history and st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # ---- Main tabs ----
    tab_analyze, tab_history, tab_about = st.tabs(["🔍 Analyze", "🕓 History", "ℹ️ About"])

    with tab_analyze:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("Interactive analysis")
            st.markdown("Select a sample prompt or enter your own text to inspect the classification.")
            sample_prompts = [
                "I hate people who are different.",
                "That comment is very rude.",
                "Have a great day everyone!",
                "He is such an idiot.",
                "This is a friendly message.",
                "I love this community.",
            ]
            render_sample_buttons(sample_prompts)
            user_text = st.text_area(
                "Enter text here",
                value=st.session_state.user_text,
                height=180,
                placeholder="Type a sentence, comment, or tweet...",
                label_visibility="collapsed",
            )
            st.session_state.user_text = user_text
            predict_clicked = st.button("Analyze text", use_container_width=True)

        with col_result:
            st.subheader("Result")
            if predict_clicked:
                if not user_text.strip():
                    st.error("Please enter some text first.")
                else:
                    label, confidence, probs = predict_text(model, user_text, class_names or {})
                    badge_text, badge_color = confidence_badge(confidence)
                    accent = CLASS_COLORS.get(label, DEFAULT_COLOR)

                    st.markdown(
                        f"""
                        <div class="result-card" style="--accent: {accent};">
                            <div class="result-label">{label}</div>
                            <div class="result-sub" style="color:{badge_color};">
                                {badge_text} · {confidence * 100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_probability_chart(probs)

                    with st.expander("See exact values"):
                        for p in probs:
                            st.write(f"- {p['Class']}: {p['Probability'] * 100:.2f}%")

                    st.session_state.history.insert(
                        0,
                        {
                            "text": user_text.strip(),
                            "label": label,
                            "confidence": confidence,
                            "time": datetime.now().strftime("%H:%M:%S"),
                        },
                    )
            else:
                st.markdown(
                    """
                    <div class="section-card">
                        <h3>How it works</h3>
                        <p>Enter text and click "Analyze text" to see category predictions, confidence, and a visual probability distribution.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_history:
        st.subheader("Recent predictions this session")
        if not st.session_state.history:
            st.caption("No predictions yet — try the Analyze tab.")
        else:
            for item in st.session_state.history:
                st.markdown(
                    f"""
                    <div class="history-item">
                        <b>{item['label']}</b> ({item['confidence']*100:.1f}%) — {item['time']}<br>
                        <span style="color:rgba(255,255,255,0.72);">
                            {item['text'][:140]}{'…' if len(item['text']) > 140 else ''}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_about:
        st.subheader("About this project")
        st.markdown(
            """
            This app classifies text into three categories using a **TF-IDF vectorizer** and **Logistic Regression**.

            - The model is trained on the Davidson hate-speech dataset.
            - It shows both the top prediction and the full probability breakdown.
            - The interface is styled for presentation and classroom demos.
            """
        )
        st.markdown("""
        **Why this app is useful for projects**
        1. Clean, presentation-ready dashboard.
        2. Real-time text prediction with confidence.
        3. Easy to deploy on Streamlit Cloud.
        """)
        st.caption("Built with Streamlit · scikit-learn · pandas · Altair")


if __name__ == "__main__":
    main()