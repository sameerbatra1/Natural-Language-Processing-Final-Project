"""
Exploratory Data Analysis for the Jigsaw Toxic Comment Classification dataset.

Follows OOP principles: each responsibility is encapsulated in its own method,
state is managed through the class, and analysis steps are composable.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Configuration dataclass — all tuneable knobs in one place
# ---------------------------------------------------------------------------

@dataclass
class EDAConfig:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    output_dir: str = "eda_outputs"
    text_col: str = "comment_text"
    id_col: str = "id"
    label_cols: list[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ])
    random_state: int = 42
    fig_dpi: int = 150
    palette: str = "Set2"


# ---------------------------------------------------------------------------
# Individual analyser classes (Single Responsibility Principle)
# ---------------------------------------------------------------------------

class NullValueAnalyser:
    """Analyses missing / null values in a DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset") -> None:
        self.df = df
        self.name = name

    def summary(self) -> pd.DataFrame:
        total = self.df.isnull().sum()
        pct = (total / len(self.df) * 100).round(4)
        return pd.DataFrame({
            "missing_count": total,
            "missing_pct": pct,
        }).sort_values("missing_count", ascending=False)

    def report(self) -> None:
        summary = self.summary()
        print(f"\n{'='*55}")
        print(f"  NULL VALUE ANALYSIS — {self.name.upper()}")
        print(f"{'='*55}")
        if summary["missing_count"].sum() == 0:
            print("  No missing values found.")
        else:
            print(summary[summary["missing_count"] > 0].to_string())
        print()


class BasicStatsAnalyser:
    """Reports shape, dtypes, and head/tail of a DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset") -> None:
        self.df = df
        self.name = name

    def report(self) -> None:
        print(f"\n{'='*55}")
        print(f"  BASIC STATISTICS — {self.name.upper()}")
        print(f"{'='*55}")
        print(f"  Shape       : {self.df.shape}")
        print(f"  Columns     : {list(self.df.columns)}")
        print(f"\n  Data types:\n{self.df.dtypes.to_string()}")
        print(f"\n  First 3 rows:\n{self.df.head(3).to_string()}")
        print()


class ClassImbalanceAnalyser:
    """
    Analyses class imbalance for multi-label classification.

    Computes per-label positive rates, multi-label co-occurrence,
    and the distribution of label cardinality (# of labels per sample).
    """

    def __init__(self, df: pd.DataFrame, label_cols: list[str]) -> None:
        self.df = df
        self.label_cols = label_cols
        self._label_df = df[label_cols]

    # ------------------------------------------------------------------
    # Per-label statistics
    # ------------------------------------------------------------------

    def label_distribution(self) -> pd.DataFrame:
        counts = self._label_df.sum()
        pct = (counts / len(self.df) * 100).round(3)
        negative = len(self.df) - counts
        return pd.DataFrame({
            "positive_count": counts,
            "negative_count": negative,
            "positive_pct": pct,
        }).sort_values("positive_pct", ascending=False)

    # ------------------------------------------------------------------
    # Label cardinality (how many labels each sample carries)
    # ------------------------------------------------------------------

    def cardinality_distribution(self) -> pd.Series:
        return self._label_df.sum(axis=1).value_counts().sort_index()

    # ------------------------------------------------------------------
    # Co-occurrence matrix
    # ------------------------------------------------------------------

    def cooccurrence_matrix(self) -> pd.DataFrame:
        return self._label_df.T.dot(self._label_df)

    def report(self) -> None:
        dist = self.label_distribution()
        print(f"\n{'='*55}")
        print("  CLASS IMBALANCE ANALYSIS")
        print(f"{'='*55}")
        print(dist.to_string())
        print(f"\n  Clean (no label) samples : "
              f"{(self._label_df.sum(axis=1) == 0).sum()} "
              f"({(self._label_df.sum(axis=1) == 0).mean()*100:.2f}%)")

        card = self.cardinality_distribution()
        print(f"\n  Label cardinality distribution (# labels per sample):")
        for n_labels, count in card.items():
            print(f"    {int(n_labels)} label(s) : {count:>7,} samples")
        print()


class TextLengthAnalyser:
    """Analyses comment length (character and word counts)."""

    def __init__(self, df: pd.DataFrame, text_col: str,
                 label_cols: Optional[list[str]] = None) -> None:
        self.df = df.copy()
        self.text_col = text_col
        self.label_cols = label_cols
        self.df["_char_len"] = self.df[text_col].str.len()
        self.df["_word_len"] = self.df[text_col].str.split().str.len()

    def overall_stats(self) -> pd.DataFrame:
        return self.df[["_char_len", "_word_len"]].describe().rename(
            columns={"_char_len": "char_count", "_word_len": "word_count"}
        )

    def stats_by_label(self) -> Optional[pd.DataFrame]:
        if not self.label_cols:
            return None
        rows = []
        for col in self.label_cols:
            for val, grp in self.df.groupby(col):
                rows.append({
                    "label": col,
                    "class": "positive" if val == 1 else "negative",
                    "mean_words": grp["_word_len"].mean().round(1),
                    "median_words": grp["_word_len"].median(),
                    "mean_chars": grp["_char_len"].mean().round(1),
                })
        return pd.DataFrame(rows).sort_values(["label", "class"])

    def report(self) -> None:
        print(f"\n{'='*55}")
        print("  TEXT LENGTH ANALYSIS")
        print(f"{'='*55}")
        print(self.overall_stats().to_string())
        by_label = self.stats_by_label()
        if by_label is not None:
            print("\n  Mean word count by label:")
            print(by_label.to_string(index=False))
        print()


class DataBiasAnalyser:
    """
    Examines potential data-level biases:
      - Duplicate comments
      - Empty / whitespace-only comments
      - Extremely short / long outliers
      - Samples with all labels positive (maximally toxic)
    """

    def __init__(self, df: pd.DataFrame, text_col: str,
                 label_cols: Optional[list[str]] = None) -> None:
        self.df = df
        self.text_col = text_col
        self.label_cols = label_cols

    def duplicate_count(self) -> int:
        return int(self.df[self.text_col].duplicated().sum())

    def empty_count(self) -> int:
        return int(self.df[self.text_col].str.strip().eq("").sum())

    def outlier_lengths(self, low_pct: float = 1.0,
                        high_pct: float = 99.0) -> dict:
        lengths = self.df[self.text_col].str.len()
        lo = np.percentile(lengths.dropna(), low_pct)
        hi = np.percentile(lengths.dropna(), high_pct)
        return {
            f"< p{int(low_pct)} ({int(lo)} chars)": int((lengths < lo).sum()),
            f"> p{int(high_pct)} ({int(hi)} chars)": int((lengths > hi).sum()),
        }

    def all_labels_positive(self) -> int:
        if not self.label_cols:
            return 0
        return int((self.df[self.label_cols].sum(axis=1)
                    == len(self.label_cols)).sum())

    def report(self) -> None:
        print(f"\n{'='*55}")
        print("  DATA BIAS / QUALITY ANALYSIS")
        print(f"{'='*55}")
        print(f"  Duplicate comments         : {self.duplicate_count():,}")
        print(f"  Empty / whitespace comments: {self.empty_count():,}")
        for label, cnt in self.outlier_lengths().items():
            print(f"  Text length outliers {label}: {cnt:,}")
        if self.label_cols:
            print(f"  All-label-positive samples : {self.all_labels_positive():,}")
        print()


# ---------------------------------------------------------------------------
# Visualiser class (Open/Closed Principle — extend by adding new plot methods)
# ---------------------------------------------------------------------------

class EDAVisualiser:
    """
    Generates and saves all EDA plots.

    Each public method corresponds to one plot; calling `plot_all` runs them
    in sequence so callers never have to know the internal order.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 config: EDAConfig) -> None:
        self.train = train_df
        self.test = test_df
        self.cfg = config
        self._out = Path(config.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"figure.dpi": config.fig_dpi})
        sns.set_theme(style="whitegrid", palette=config.palette)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self._out / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_label_distribution(self) -> None:
        """Bar chart of positive sample counts per label."""
        label_cols = self.cfg.label_cols
        counts = self.train[label_cols].sum().sort_values(ascending=False)
        total = len(self.train)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute counts
        ax = axes[0]
        bars = ax.bar(counts.index, counts.values,
                      color=sns.color_palette(self.cfg.palette, len(label_cols)))
        ax.set_title("Positive Sample Count per Label", fontsize=13, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 200,
                    f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # Percentage
        ax2 = axes[1]
        pcts = (counts / total * 100).values
        bars2 = ax2.bar(counts.index, pcts,
                        color=sns.color_palette(self.cfg.palette, len(label_cols)))
        ax2.set_title("Positive Sample % per Label", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Label")
        ax2.set_ylabel("Percentage (%)")
        for bar, pct in zip(bars2, pcts):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{pct:.2f}%",
                     ha="center", va="bottom", fontsize=8)
        ax2.tick_params(axis="x", rotation=30)

        fig.suptitle("Class Imbalance — Jigsaw Train Set", fontsize=15, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "01_label_distribution")

    def plot_label_cardinality(self) -> None:
        """How many labels does each comment carry?"""
        label_cols = self.cfg.label_cols
        cardinality = self.train[label_cols].sum(axis=1).value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = sns.color_palette(self.cfg.palette, len(cardinality))
        bars = ax.bar(cardinality.index.astype(str), cardinality.values, color=colors)
        ax.set_title("Label Cardinality Distribution\n(# toxic labels per comment)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Labels")
        ax.set_ylabel("Sample Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 100,
                    f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        self._save(fig, "02_label_cardinality")

    def plot_cooccurrence_heatmap(self) -> None:
        """Heatmap of label co-occurrence matrix."""
        label_cols = self.cfg.label_cols
        co = self.train[label_cols].T.dot(self.train[label_cols])

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.zeros_like(co, dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(co, annot=True, fmt=",d", cmap="YlOrRd",
                    mask=mask, ax=ax, linewidths=0.5)
        ax.set_title("Label Co-occurrence Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "03_cooccurrence_heatmap")

    def plot_text_length_distributions(self) -> None:
        """Overlapping KDE plots of word-count distributions across labels."""
        df = self.train.copy()
        df["_word_len"] = df[self.cfg.text_col].str.split().str.len()
        label_cols = self.cfg.label_cols

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
        axes = axes.flatten()

        palette = sns.color_palette(self.cfg.palette, 2)
        for ax, col in zip(axes, label_cols):
            pos = df.loc[df[col] == 1, "_word_len"].clip(upper=500)
            neg = df.loc[df[col] == 0, "_word_len"].clip(upper=500)
            sns.kdeplot(neg, ax=ax, label="Negative", fill=True,
                        color=palette[0], alpha=0.5, warn_singular=False)
            sns.kdeplot(pos, ax=ax, label="Positive", fill=True,
                        color=palette[1], alpha=0.6, warn_singular=False)
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("Word count (capped at 500)")
            ax.legend(fontsize=8)

        fig.suptitle("Word Count Distributions: Positive vs Negative per Label",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "04_text_length_by_label")

    def plot_char_length_boxplot(self) -> None:
        """Box-plot of character-level length by label (positive vs negative)."""
        df = self.train.copy()
        df["_char_len"] = df[self.cfg.text_col].str.len()
        label_cols = self.cfg.label_cols

        rows = []
        for col in label_cols:
            for val, grp in df.groupby(col):
                for length in grp["_char_len"].sample(min(5000, len(grp)),
                                                       random_state=self.cfg.random_state):
                    rows.append({"label": col,
                                 "class": "positive" if val == 1 else "negative",
                                 "char_len": length})
        plot_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=plot_df, x="label", y="char_len", hue="class",
                    palette=self.cfg.palette, ax=ax, showfliers=False)
        ax.set_title("Character Length by Label Class (outliers hidden)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel("Character Count")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        self._save(fig, "05_char_length_boxplot")

    def plot_train_test_size_comparison(self) -> None:
        """Bar chart comparing train vs test set sizes."""
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = {"Train": len(self.train), "Test": len(self.test)}
        colors = sns.color_palette(self.cfg.palette, 2)
        bars = ax.bar(sizes.keys(), sizes.values(), color=colors, width=0.4)
        ax.set_title("Train / Test Set Sizes", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Samples")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1000,
                    f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        self._save(fig, "06_train_test_sizes")

    def plot_null_heatmap(self) -> None:
        """Heatmap of null values (sampled for speed)."""
        sample = self.train.sample(min(5000, len(self.train)),
                                   random_state=self.cfg.random_state)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(sample.isnull(), cbar=False, yticklabels=False,
                    cmap="viridis", ax=ax)
        ax.set_title("Null Value Map — Train (5 k row sample)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "07_null_heatmap")

    def plot_wordclouds(self, max_words: int = 150) -> None:
        """Word cloud for toxic vs clean comments."""
        text_col = self.cfg.text_col

        toxic_mask = self.train[self.cfg.label_cols].sum(axis=1) > 0
        toxic_text = " ".join(self.train.loc[toxic_mask, text_col].dropna()
                               .sample(min(5000, toxic_mask.sum()),
                                       random_state=self.cfg.random_state))
        clean_text = " ".join(self.train.loc[~toxic_mask, text_col].dropna()
                               .sample(min(5000, (~toxic_mask).sum()),
                                       random_state=self.cfg.random_state))

        wc_toxic = WordCloud(width=800, height=400, background_color="black",
                             colormap="Reds", max_words=max_words,
                             collocations=False).generate(toxic_text)
        wc_clean = WordCloud(width=800, height=400, background_color="white",
                             colormap="Blues", max_words=max_words,
                             collocations=False).generate(clean_text)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(wc_toxic, interpolation="bilinear")
        axes[0].axis("off")
        axes[0].set_title("Toxic Comments", fontsize=13, fontweight="bold")

        axes[1].imshow(wc_clean, interpolation="bilinear")
        axes[1].axis("off")
        axes[1].set_title("Clean Comments", fontsize=13, fontweight="bold")

        fig.suptitle("Word Clouds: Toxic vs Clean", fontsize=15, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "08_wordclouds")

    def plot_correlation_matrix(self) -> None:
        """Pearson correlation between label columns."""
        label_cols = self.cfg.label_cols
        corr = self.train[label_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title("Label Correlation Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "09_label_correlation")

    def plot_all(self) -> None:
        """Run every plot method in sequence."""
        plot_methods = [
            self.plot_label_distribution,
            self.plot_label_cardinality,
            self.plot_cooccurrence_heatmap,
            self.plot_text_length_distributions,
            self.plot_char_length_boxplot,
            self.plot_train_test_size_comparison,
            self.plot_null_heatmap,
            self.plot_wordclouds,
            self.plot_correlation_matrix,
        ]
        print(f"\n  Saving plots to '{self.cfg.output_dir}/' ...\n")
        for method in plot_methods:
            try:
                method()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {method.__name__} failed: {exc}")


# ---------------------------------------------------------------------------
# Orchestrator class (Facade / Single entry-point)
# ---------------------------------------------------------------------------

class JigsawEDA:
    """
    Top-level orchestrator for Jigsaw dataset EDA.

    Usage
    -----
    >>> eda = JigsawEDA()
    >>> eda.run()               # run everything
    >>> eda.run(plots=False)    # text reports only
    """

    def __init__(self, config: Optional[EDAConfig] = None) -> None:
        self.cfg = config or EDAConfig()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        print(f"Loading train data from '{self.cfg.train_path}' …")
        self.train_df = pd.read_csv(self.cfg.train_path)
        print(f"Loading test  data from '{self.cfg.test_path}'  …")
        self.test_df = pd.read_csv(self.cfg.test_path)
        print(f"  Train shape : {self.train_df.shape}")
        print(f"  Test  shape : {self.test_df.shape}")

    # ------------------------------------------------------------------
    # Report helpers (compose the individual analysers)
    # ------------------------------------------------------------------

    def _run_text_reports(self) -> None:
        assert self.train_df is not None and self.test_df is not None, \
            "Call load_data() first."

        # Basic stats
        BasicStatsAnalyser(self.train_df, "train").report()
        BasicStatsAnalyser(self.test_df, "test").report()

        # Null values
        NullValueAnalyser(self.train_df, "train").report()
        NullValueAnalyser(self.test_df, "test").report()

        # Class imbalance (train only — test has no labels)
        ClassImbalanceAnalyser(self.train_df, self.cfg.label_cols).report()

        # Text length
        TextLengthAnalyser(
            self.train_df, self.cfg.text_col, self.cfg.label_cols
        ).report()

        # Data bias / quality
        DataBiasAnalyser(
            self.train_df, self.cfg.text_col, self.cfg.label_cols
        ).report()
        DataBiasAnalyser(
            self.test_df, self.cfg.text_col
        ).report()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, plots: bool = True) -> None:
        """Execute the full EDA pipeline."""
        self.load_data()
        self._run_text_reports()
        if plots:
            vis = EDAVisualiser(self.train_df, self.test_df, self.cfg)
            vis.plot_all()
        print("\nEDA complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = EDAConfig(
        train_path="data/train.csv",
        test_path="data/test.csv",
        output_dir="eda_outputs",
    )
    eda = JigsawEDA(config=config)
    eda.run(plots=True)
