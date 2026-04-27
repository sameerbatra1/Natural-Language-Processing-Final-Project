from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    alpha: Optional[list[float]] = None
    reduction: str = "mean"
    label_cols: list[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ])


# ---------------------------------------------------------------------------
# Alpha computer
# ---------------------------------------------------------------------------

class AlphaComputer:
    def __init__(self, df: pd.DataFrame, label_cols: list[str]) -> None:
        self._df = df
        self._label_cols = label_cols

    def compute(self) -> torch.Tensor:
        n_total = len(self._df)
        alphas = []
        for col in self._label_cols:
            n_pos = self._df[col].sum()
            n_neg = n_total - n_pos
            alphas.append(n_neg / n_total)
        return torch.tensor(alphas, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(
        self,
        config: FocalLossConfig,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = config.gamma
        self.reduction = config.reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        elif config.alpha is not None:
            self.register_buffer(
                "alpha", torch.tensor(config.alpha, dtype=torch.float32)
            )
        else:
            self.register_buffer("alpha", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Entry point — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    label_cols = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ]

    print("Loading train data ...")
    df = pd.read_csv("data/train.csv")

    alpha_tensor = AlphaComputer(df, label_cols).compute()
    print("\nComputed alpha per label:")
    for col, a in zip(label_cols, alpha_tensor.tolist()):
        print(f"  {col:<15} {a:.4f}")

    config = FocalLossConfig(gamma=2.0, reduction="mean", label_cols=label_cols)
    criterion = FocalLoss(config=config, alpha=alpha_tensor)

    batch_size = 8
    torch.manual_seed(42)
    dummy_logits  = torch.randn(batch_size, len(label_cols))
    dummy_targets = torch.randint(0, 2, (batch_size, len(label_cols))).float()

    loss = criterion(dummy_logits, dummy_targets)
    print(f"\nForward pass loss (batch={batch_size}, gamma={config.gamma}): {loss.item():.6f}")

    print("\nReduction='none' output shape:", criterion.__class__.__name__)
    config_none = FocalLossConfig(gamma=2.0, reduction="none", label_cols=label_cols)
    criterion_none = FocalLoss(config=config_none, alpha=alpha_tensor)
    per_element = criterion_none(dummy_logits, dummy_targets)
    print(f"  per-element shape : {per_element.shape}   (expected: [{batch_size}, {len(label_cols)}])")

    print("\nSanity check passed.")
