from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_constant_schedule_with_warmup


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_every_n_steps: int = 100
    label_cols: list[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ])
    # ReduceLROnPlateau
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-7
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    results_csv: str = "training_results.csv"
    # Checkpoint / ReduceLROnPlateau / early stopping maximize this validation key
    monitor_metric: Literal["mean_auc", "mean_f1"] = "mean_f1"

    def __post_init__(self) -> None:
        if self.monitor_metric not in ("mean_auc", "mean_f1"):
            raise ValueError(
                "monitor_metric must be 'mean_auc' or 'mean_f1', "
                f"got {self.monitor_metric!r}"
            )


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    def __init__(self, label_cols: list[str]) -> None:
        self._label_cols = label_cols
        self._all_logits: list[np.ndarray] = []
        self._all_targets: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._all_logits.append(torch.sigmoid(logits).detach().cpu().numpy())
        self._all_targets.append(targets.detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        preds   = np.concatenate(self._all_logits,  axis=0)
        targets = np.concatenate(self._all_targets, axis=0)
        metrics: dict[str, float] = {}
        aucs = []
        f1s = []
        for i, col in enumerate(self._label_cols):
            if targets[:, i].sum() == 0:
                metrics[col] = float("nan")
                metrics[f"{col}_f1"] = float("nan")
            else:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                metrics[col] = round(auc, 4)
                aucs.append(auc)
                y_bin = (preds[:, i] >= 0.5).astype(np.int32)
                f1 = f1_score(targets[:, i], y_bin, zero_division=0)
                metrics[f"{col}_f1"] = round(float(f1), 4)
                f1s.append(f1)
        metrics["mean_auc"] = round(float(np.mean(aucs)), 4) if aucs else float("nan")
        metrics["mean_f1"] = round(float(np.mean(f1s)), 4) if f1s else float("nan")
        return metrics

    def reset(self) -> None:
        self._all_logits.clear()
        self._all_targets.clear()


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._best_metric = -float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        metric_name: str,
    ) -> bool:
        if metric > self._best_metric:
            self._best_metric = metric
            path = self._dir / "best_model.pt"
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "metric":       metric,
                "metric_name":  metric_name,
            }, path)
            print(f"  Checkpoint saved → {path}  ({metric_name}={metric:.4f})")
            return True
        return False

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        path = self._dir / "best_model.pt"
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        mname = ckpt.get("metric_name", "metric")
        print(f"  Loaded checkpoint from epoch {ckpt['epoch']}  ({mname}={ckpt['metric']:.4f})")
        return ckpt


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when the monitored validation metric fails to improve for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self._patience  = patience
        self._min_delta = min_delta
        self._best      = -float("inf")
        self._counter   = 0

    @property
    def counter(self) -> int:
        return self._counter

    def step(self, metric: float) -> bool:
        """Return ``True`` when training should stop."""
        if metric > self._best + self._min_delta:
            self._best   = metric
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self._patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainerConfig,
    ) -> None:
        self._model      = model.to(config.device)
        self._train_dl   = train_loader
        self._val_dl     = val_loader
        self._criterion  = criterion
        self._cfg        = config
        self._device     = config.device
        self._ckpt_mgr   = CheckpointManager(config.checkpoint_dir)
        self._metrics    = MetricsTracker(config.label_cols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self._device) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Single epoch passes
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        self._model.train()
        total_loss = 0.0
        step = 0

        bar = tqdm(self._train_dl, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in bar:
            batch    = self._to_device(batch)
            logits   = self._model(batch["input_ids"], batch["attention_mask"])
            loss     = self._criterion(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1
            if step % self._cfg.log_every_n_steps == 0:
                bar.set_postfix(loss=f"{total_loss / step:.4f}")

        return total_loss / len(self._train_dl)

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self._model.eval()
        self._metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            bar = tqdm(self._val_dl, desc=f"Epoch {epoch} [val]  ", leave=False)
            for batch in bar:
                batch   = self._to_device(batch)
                logits  = self._model(batch["input_ids"], batch["attention_mask"])
                loss    = self._criterion(logits, batch["labels"])
                total_loss += loss.item()
                self._metrics.update(logits, batch["labels"])

        metrics = self._metrics.compute()
        metrics["val_loss"] = round(total_loss / len(self._val_dl), 4)
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        total_steps  = len(self._train_dl) * self._cfg.epochs
        warmup_steps = int(total_steps * self._cfg.warmup_ratio)

        optimizer = AdamW(
            self._model.parameters(),
            lr=self._cfg.lr,
            weight_decay=self._cfg.weight_decay,
        )

        # Step-level scheduler: linear warmup then constant LR.
        # ReduceLROnPlateau handles decay from there, so we avoid the two
        # schedulers fighting each other.
        step_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self._cfg.reduce_lr_factor,
            patience=self._cfg.reduce_lr_patience,
            min_lr=self._cfg.reduce_lr_min_lr,
        )
        early_stopper = EarlyStopping(
            patience=self._cfg.early_stopping_patience,
            min_delta=self._cfg.early_stopping_min_delta,
        )
        history: list[dict] = []

        mon = self._cfg.monitor_metric
        print(f"\nTraining on {self._device}  |  "
              f"{self._cfg.epochs} epochs  |  "
              f"{total_steps:,} total steps  |  "
              f"{warmup_steps} warmup steps  |  "
              f"monitor={mon}\n")

        for epoch in range(1, self._cfg.epochs + 1):
            t0 = time.time()
            train_loss  = self._train_epoch(optimizer, step_scheduler, epoch)
            val_metrics = self._val_epoch(epoch)
            elapsed     = time.time() - t0

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{self._cfg.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_metrics['val_loss']:.4f}  "
                  f"mean_auc={val_metrics['mean_auc']:.4f}  "
                  f"mean_f1={val_metrics['mean_f1']:.4f}  "
                  f"lr={current_lr:.2e}  "
                  f"({elapsed:.0f}s)")

            for col in self._cfg.label_cols:
                auc = val_metrics.get(col, float("nan"))
                f1v = val_metrics.get(f"{col}_f1", float("nan"))
                print(f"  {col:<16} auc={auc:.4f}  f1={f1v:.4f}")

            row: dict = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": val_metrics["val_loss"],
                "mean_auc": val_metrics["mean_auc"],
                "mean_f1": val_metrics["mean_f1"],
            }
            for col in self._cfg.label_cols:
                row[f"{col}_auc"] = val_metrics[col]
                row[f"{col}_f1"] = val_metrics[f"{col}_f1"]
            history.append(row)

            monitor_score = float(val_metrics[mon])
            self._ckpt_mgr.save(
                self._model, optimizer, epoch, monitor_score, mon,
            )

            plateau_scheduler.step(monitor_score)

            if early_stopper.step(monitor_score):
                print(f"\nEarly stopping triggered on {mon} — no improvement for "
                      f"{self._cfg.early_stopping_patience} epochs.")
                break

        out_path = Path(self._cfg.results_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(out_path, index=False)
        print(f"\nTraining complete. Metrics saved to {out_path.resolve()}.")
