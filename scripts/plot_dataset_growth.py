#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bar chart of the per-class growth from splits-v2 (HAM10000 only) to
splits-v3 (HAM10000 + BCN20000).

Each class gets one horizontal stacked bar: the HAM10000 images
(identical to splits-v2) plus the BCN20000 images added in splits-v3.
The label at the end of each bar gives the splits-v3 total and the
relative increase.

Usage (from the project root):
    python3 scripts/plot_dataset_growth.py
    python3 scripts/plot_dataset_growth.py --subset training
"""

import argparse

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_V3 = PROJECT_ROOT / "data" / "splits-v3"

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "figures" / "splits_v3_class_growth.png"

CLASS_NAMES = {
    "nv": "nv — Melanocytic nevus",
    "mel": "mel — Melanoma",
    "bcc": "bcc — Basal cell carcinoma",
    "bkl": "bkl — Benign keratosis",
    "akiec": "akiec — Actinic keratosis / SCC",
    "vasc": "vasc — Vascular lesion",
    "df": "df — Dermatofibroma",
}

# Validated categorical palette (light surface)

SURFACE = "#fcfcfb"
HAM_COLOR = "#2a78d6"
BCN_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_counts(subset):

    """
    Return a dataframe indexed by class with HAM10000 and BCN20000 counts.
    """

    subsets = ["training", "validation", "test"] if subset == "all" else [subset]

    frames = [
        pd.read_csv(SPLITS_V3 / f"{name}_metadata.csv")
        for name in subsets
    ]

    metadata = pd.concat(frames, ignore_index=True)

    counts = (
        metadata
        .groupby(["dx", "source"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["HAM10000", "BCN20000"], fill_value=0)
    )

    counts["total"] = counts.sum(axis=1)

    return counts.sort_values("total")


def plot(counts, subset, output_path):

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels = [CLASS_NAMES.get(c, c) for c in counts.index]
    y = range(len(counts))
    height = 0.62

    # White edge = 2 px surface gap between the two stacked segments

    ax.barh(
        y, counts["HAM10000"], height=height, color=HAM_COLOR,
        edgecolor=SURFACE, linewidth=1.5,
        label="HAM10000 (= splits-v2)"
    )
    ax.barh(
        y, counts["BCN20000"], left=counts["HAM10000"], height=height,
        color=BCN_COLOR, edgecolor=SURFACE, linewidth=1.5,
        label="BCN20000 (added in splits-v3)"
    )

    x_max = counts["total"].max()

    for i, (ham, bcn, total) in enumerate(
        counts[["HAM10000", "BCN20000", "total"]].itertuples(index=False)
    ):
        increase = 100 * bcn / ham if ham else float("inf")
        # Offsets in points, so labels never collide whatever the scale
        ax.annotate(
            f"{total:,}", (total, i), xytext=(5, 0),
            textcoords="offset points",
            va="center", ha="left", fontsize=9.5,
            color=TEXT_PRIMARY, fontweight="bold"
        )
        ax.annotate(
            f"{ham:,} + {bcn:,}  (+{increase:.0f} %)", (total, i),
            xytext=(52, 0), textcoords="offset points",
            va="center", ha="left", fontsize=8.5, color=TEXT_SECONDARY
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_PRIMARY)
    ax.set_xlim(0, x_max * 1.40)
    ax.set_xlabel("Images", fontsize=9.5, color=TEXT_SECONDARY)
    ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ham_total = int(counts["HAM10000"].sum())
    v3_total = int(counts["total"].sum())
    scope = "all subsets" if subset == "all" else f"{subset} subset"

    ax.set_title(
        f"Images per class: splits-v2 → splits-v3 ({scope})\n",
        loc="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold"
    )
    ax.text(
        0, 1.02,
        f"{ham_total:,} → {v3_total:,} images "
        f"(+{100 * (v3_total - ham_total) / ham_total:.0f} %). "
        "Labels: splits-v3 total, HAM10000 + BCN20000 (relative increase).",
        transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY
    )
    ax.legend(
        loc="lower right", frameon=False, fontsize=9,
        labelcolor=TEXT_PRIMARY
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--subset",
        choices=["all", "training", "validation", "test"],
        default="all"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    counts = load_counts(args.subset)
    print(counts.to_string())
    plot(counts, args.subset, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
