"""
inference.py — Tokenisation + model inference pipeline for the Jigsaw app.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, MAX_LENGTH, MODEL_NAME


class ToxicityPredictor:
    """Loads the model once and exposes a simple predict() method."""

    def __init__(self, model_name: str = MODEL_NAME, max_len: int = MAX_LENGTH) -> None:
        self._device = torch.device("cpu")   # CPU-only for Streamlit / HF Spaces
        self._max_len = max_len
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, float]:
        """
        Tokenise *text*, run inference, apply sigmoid, and return
        a dict mapping each toxicity label → probability (0-1).
        """
        text = text.strip() or "[EMPTY]"

        encoding = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_len,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids=input_ids,
                                 attention_mask=attention_mask).logits

        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        # unitary/toxic-bert outputs exactly 6 logits matching LABELS order
        if len(probs) != len(LABELS):
            # fallback: pad / truncate to match LABELS length
            probs = (probs + [0.0] * len(LABELS))[:len(LABELS)]

        return {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
