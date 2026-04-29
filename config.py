"""
config.py — Central configuration for the Jigsaw Streamlit app.
"""

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

LABEL_DISPLAY_NAMES = {
    "toxic":         "☣️  Toxic",
    "severe_toxic":  "💀 Severe Toxic",
    "obscene":       "🤬 Obscene",
    "threat":        "⚠️  Threat",
    "insult":        "😤 Insult",
    "identity_hate": "🎯 Identity Hate",
}

# Hugging Face model — works out-of-the-box, no local checkpoint needed.
# Swap to your own fine-tuned model path/repo if you have one.
MODEL_NAME = "unitary/toxic-bert"

MAX_LENGTH = 256
DEFAULT_THRESHOLD = 0.5

# Colour bands for probability bars
def severity_color(prob: float) -> str:
    if prob >= 0.75:
        return "#e63946"   # red
    if prob >= 0.50:
        return "#f4a261"   # orange
    if prob >= 0.25:
        return "#e9c46a"   # yellow
    return "#52b788"       # green
