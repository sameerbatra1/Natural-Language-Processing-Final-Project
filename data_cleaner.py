from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CleaningConfig:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    output_dir: str = "data"
    text_col: str = "comment_text"
    id_col: str = "id"
    label_cols: list[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ])
    empty_placeholder: str = "[EMPTY]"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete cleaners
# ---------------------------------------------------------------------------

class HTMLStripper(BaseCleaner):
    def clean(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text()


class WikiMarkupCleaner(BaseCleaner):
    _wikilink = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
    _template = re.compile(r"\{\{[^}]*\}\}")
    _heading  = re.compile(r"={2,}[^=]+=+")

    def clean(self, text: str) -> str:
        text = self._wikilink.sub(r"\1", text)
        text = self._template.sub("", text)
        text = self._heading.sub("", text)
        return text


class URLAnonymiser(BaseCleaner):
    _url = re.compile(r"https?://\S+|www\.\S+")

    def clean(self, text: str) -> str:
        return self._url.sub("[URL]", text)


class IPAnonymiser(BaseCleaner):
    _ipv4 = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )

    def clean(self, text: str) -> str:
        return self._ipv4.sub("[IP]", text)


class UsernameAnonymiser(BaseCleaner):
    # [[User:Name|display]] or [[User:Name]]
    _wikilink_user = re.compile(r"\[\[User:[^\]|]+(?:\|[^\]]*)?\]\]", re.IGNORECASE)
    # bare User:Name (not inside brackets)
    _bare_user = re.compile(r"\bUser:[A-Za-z0-9_\-\./ ]+", re.IGNORECASE)
    # @Name: mention
    _mention = re.compile(r"@[A-Za-z0-9_\-\.]+:?")
    # talk-page signature: word-chars followed by (talk) with optional timestamp
    _talk_sig = re.compile(
        r"[A-Za-z0-9_\-\. ]+\s*\(talk\)(?:\s*\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\))?",
        re.IGNORECASE,
    )

    def clean(self, text: str) -> str:
        text = self._wikilink_user.sub("[USERNAME]", text)
        text = self._bare_user.sub("[USERNAME]", text)
        text = self._mention.sub("[USERNAME]", text)
        text = self._talk_sig.sub("[USERNAME]", text)
        return text


class WhitespaceNormalizer(BaseCleaner):
    _multi_space = re.compile(r"[ \t\r\n]+")

    def clean(self, text: str) -> str:
        return self._multi_space.sub(" ", text).strip()


class UnicodeNormalizer(BaseCleaner):
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Cc")


class EmptyTextHandler(BaseCleaner):
    def __init__(self, placeholder: str = "[EMPTY]") -> None:
        self.placeholder = placeholder

    def clean(self, text: str) -> str:
        return text if text.strip() else self.placeholder


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CleaningPipeline:
    def __init__(self, cleaners: list[BaseCleaner]) -> None:
        self._cleaners = cleaners

    def run(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        for cleaner in self._cleaners:
            text = cleaner.clean(text)
        return text

    def run_on_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.run)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataCleaner:
    def __init__(self, config: Optional[CleaningConfig] = None) -> None:
        self.cfg = config or CleaningConfig()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        print(f"Loading '{self.cfg.train_path}' ...")
        self.train_df = pd.read_csv(self.cfg.train_path)
        print(f"Loading '{self.cfg.test_path}'  ...")
        self.test_df = pd.read_csv(self.cfg.test_path)
        print(f"  Train : {self.train_df.shape}  |  Test : {self.test_df.shape}")

    def build_pipeline(self) -> CleaningPipeline:
        return CleaningPipeline([
            HTMLStripper(),
            WikiMarkupCleaner(),
            URLAnonymiser(),
            IPAnonymiser(),
            UsernameAnonymiser(),
            WhitespaceNormalizer(),
            UnicodeNormalizer(),
            EmptyTextHandler(self.cfg.empty_placeholder),
        ])

    def clean(self) -> None:
        assert self.train_df is not None and self.test_df is not None, \
            "Call load_data() first."
        pipeline = self.build_pipeline()
        col = self.cfg.text_col
        print("Cleaning train ...")
        self.train_df[col] = pipeline.run_on_series(self.train_df[col])
        print("Cleaning test  ...")
        self.test_df[col] = pipeline.run_on_series(self.test_df[col])
        print("Cleaning complete.")

    def save(self) -> None:
        assert self.train_df is not None and self.test_df is not None, \
            "Call clean() first."
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_out = out / "train_clean.csv"
        test_out  = out / "test_clean.csv"
        self.train_df.to_csv(train_out, index=False)
        self.test_df.to_csv(test_out,  index=False)
        print(f"Saved → {train_out}")
        print(f"Saved → {test_out}")

    def run(self) -> None:
        self.load_data()
        self.clean()
        self.save()
        print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = CleaningConfig(
        train_path="data/train.csv",
        test_path="data/test.csv",
        output_dir="data",
    )
    DataCleaner(config=config).run()
