from __future__ import annotations

import pandas as pd

from dataset import DataConfig, DataLoaderFactory
from focal_loss import AlphaComputer, FocalLoss, FocalLossConfig
from model import JigsawBERTClassifier, ModelConfig
from trainer import Trainer, TrainerConfig

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------------------------------------------------------------
# Configs — edit these to tune the run
# ---------------------------------------------------------------------------

data_cfg = DataConfig(
    train_path="data/train_clean.csv",
    test_path="data/test_clean.csv",
    model_name="bert-base-uncased",
    max_len=256,
    batch_size=32,
    val_size=0.15,
    random_state=42,
    num_workers=2,
)

model_cfg = ModelConfig(
    model_name="bert-base-uncased",
    num_labels=len(LABEL_COLS),
    dropout=0.3,
    freeze_bert_layers=0,
)

trainer_cfg = TrainerConfig(
    epochs=5,
    lr=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    checkpoint_dir="checkpoints",
    log_every_n_steps=100,
    label_cols=LABEL_COLS,
)

# ---------------------------------------------------------------------------
# Build loaders
# ---------------------------------------------------------------------------

factory = DataLoaderFactory(data_cfg)
train_loader, val_loader, test_loader = factory.build()

# ---------------------------------------------------------------------------
# Build loss — alpha computed from the train split only
# ---------------------------------------------------------------------------

train_df = factory.get_train_df()
alpha    = AlphaComputer(train_df, LABEL_COLS).compute()
criterion = FocalLoss(FocalLossConfig(gamma=2.0, reduction="mean"), alpha=alpha)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

model = JigsawBERTClassifier(model_cfg)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

Trainer(model, train_loader, val_loader, criterion, trainer_cfg).train()
