from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    train_path: str = "data/train_clean.csv"
    test_path: str = "data/test_clean.csv"
    text_col: str = "comment_text"
    label_cols: list[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ])
    model_name: str = "bert-base-uncased"
    max_len: int = 256
    batch_size: int = 32
    val_size: float = 0.15
    random_state: int = 42
    num_workers: int = 2


# ---------------------------------------------------------------------------
# Data splitter
# ---------------------------------------------------------------------------

class DataSplitter:
    def __init__(self, df: pd.DataFrame, config: DataConfig) -> None:
        self._df = df
        self._cfg = config

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = train_test_split(
            self._df,
            test_size=self._cfg.val_size,
            stratify=self._df["toxic"],
            random_state=self._cfg.random_state,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JigsawDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizerFast,
        config: DataConfig,
        is_test: bool = False,
    ) -> None:
        self._texts = df[config.text_col].tolist()
        self._tokenizer = tokenizer
        self._max_len = config.max_len
        self._is_test = is_test
        if not is_test:
            self._labels = df[config.label_cols].values.astype("float32")

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self._tokenizer(
            self._texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self._max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if not self._is_test:
            item["labels"] = torch.tensor(self._labels[idx], dtype=torch.float32)
        else:
            item["labels"] = torch.zeros(6, dtype=torch.float32)
        return item


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

class DataLoaderFactory:
    def __init__(self, config: DataConfig) -> None:
        self._cfg = config

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        tokenizer = BertTokenizerFast.from_pretrained(self._cfg.model_name)

        train_full = pd.read_csv(self._cfg.train_path)
        test_df    = pd.read_csv(self._cfg.test_path)

        train_df, val_df = DataSplitter(train_full, self._cfg).split()

        train_ds = JigsawDataset(train_df, tokenizer, self._cfg, is_test=False)
        val_ds   = JigsawDataset(val_df,   tokenizer, self._cfg, is_test=False)
        test_ds  = JigsawDataset(test_df,  tokenizer, self._cfg, is_test=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=self._cfg.batch_size,
            shuffle=True,
            num_workers=self._cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=self._cfg.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=self._cfg.num_workers,
            pin_memory=True,
        )

        print(
            f"Splits — train: {len(train_ds):,}  "
            f"val: {len(val_ds):,}  "
            f"test: {len(test_ds):,}"
        )
        return train_loader, val_loader, test_loader

    def get_train_df(self) -> pd.DataFrame:
        train_full = pd.read_csv(self._cfg.train_path)
        train_df, _ = DataSplitter(train_full, self._cfg).split()
        return train_df
