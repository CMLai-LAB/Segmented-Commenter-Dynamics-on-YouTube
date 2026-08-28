#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT_CSV = Path("emotion_stat_tests_by_media_type/tier_standardized_residuals_by_media_type.csv")
OUTPUT_DIR = Path("emotion_stat_tests_by_media_type")
OUTPUT_COMBINED = OUTPUT_DIR / "tier_emotion_standardized_residual_forest_by_media_type.png"
OUTPUT_TRADITIONAL = OUTPUT_DIR / "tier_emotion_standardized_residual_forest_traditional.png"
OUTPUT_EMERGING = OUTPUT_DIR / "tier_emotion_standardized_residual_forest_emerging.png"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
MEDIA_TYPE_LABELS = {
    "traditional": "Traditional Media",
    "emerging": "Digital / Emerging Media",
}
TIER_ORDER = ["tier_2_5", "tier_6_20", "tier_21_50", "tier_51_plus"]
TIER_LABELS = {
    "tier_2_5": "2-5 comments",
    "tier_6_20": "6-20 comments",
    "tier_21_50": "21-50 comments",
    "tier_51_plus": "51+ comments",
}
TIER_MARKERS = {
    "tier_2_5": "o",
    "tier_6_20": "^",
    "tier_21_50": "D",
    "tier_51_plus": "s",
}
TIER_OFFSETS = {
    "tier_2_5": -0.24,
    "tier_6_20": -0.08,
    "tier_21_50": 0.08,
    "tier_51_plus": 0.24,
}
EMOTION_ORDER = ["Neutral", "Concerned", "Happy", "Angry", "Sad", "Questioning", "Surprised", "Disgusted"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tier-emotion standardized residual forests by media type.")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--significance-cutoff", type=float, default=1.96)
    return parser.parse_args()


def configure_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft JhengHei",
        "PingFang TC",
        "Heiti TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_rows(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            media_type = row["media_type"]
            tier = row["row"]
            emotion = row["emotion"]
            rows[(media_type, tier, emotion)] = {
                "observed": float(row["observed"]),
                "expected": float(row["expected"]),
                "residual": float(row["standardized_residual"]),
            }
    return rows


def residual_limit(rows, cutoff: float):
    residual_values = [item["residual"] for item in rows.values()]
    max_abs = max(abs(value) for value in residual_values) if residual_values else cutoff
    return max(cutoff + 0.8, max_abs + 0.9)


def draw_forest(ax, rows, media_type: str, cutoff: float, limit: float, show_ylabel: bool = True):
    positive_color = "#2F7D7E"
    negative_color = "#B2554A"
    neutral_color = "#A5A5A5"
    y_positions = {emotion: idx for idx, emotion in enumerate(reversed(EMOTION_ORDER))}

    ax.axvspan(-cutoff, cutoff, color="#f7f7f7", zorder=0)
    ax.axvline(0, color="#8f8f8f", linestyle="-", linewidth=1.0, zorder=1)
    ax.axvline(cutoff, color="#c8c8c8", linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(-cutoff, color="#c8c8c8", linestyle=":", linewidth=1.0, zorder=1)

    for emotion in EMOTION_ORDER:
        base_y = y_positions[emotion]
        for tier in TIER_ORDER:
            item = rows.get((media_type, tier, emotion))
            if item is None:
                continue
            residual = item["residual"]
            y = base_y + TIER_OFFSETS[tier]
            significant = abs(residual) >= cutoff
            color = positive_color if residual > 0 else negative_color if residual < 0 else neutral_color
            alpha = 1.0 if significant else 0.45
            marker_size = 54 if significant else 38
            ax.plot(
                [0, residual],
                [y, y],
                color=color,
                linewidth=2.4 if significant else 1.5,
                alpha=alpha,
                zorder=2,
            )
            ax.scatter(
                residual,
                y,
                marker=TIER_MARKERS[tier],
                s=marker_size,
                facecolors=color if significant else "white",
                edgecolors=color,
                linewidths=1.25,
                alpha=alpha,
                zorder=3,
            )

    ax.set_yticks([y_positions[emotion] for emotion in reversed(EMOTION_ORDER)])
    if show_ylabel:
        ax.set_yticklabels(list(reversed(EMOTION_ORDER)), fontsize=11, color="black")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=10, colors="black")
    ax.tick_params(axis="y", length=0, colors="black")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.9)
    ax.grid(axis="y", color="#f2f2f2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Standardized residual", fontsize=10.5, color="black")
    ax.set_title(MEDIA_TYPE_LABELS[media_type], fontsize=14, fontweight="bold", color="black", pad=14)
    ax.text(cutoff, len(EMOTION_ORDER) - 0.3, "+1.96", ha="center", va="bottom", fontsize=8.8, color="#000000")
    ax.text(-cutoff, len(EMOTION_ORDER) - 0.3, "-1.96", ha="center", va="bottom", fontsize=8.8, color="#000000")
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.9)


def legend_handles():
    positive_color = "#2F7D7E"
    negative_color = "#B2554A"
    handles = [
        Line2D(
            [0],
            [0],
            marker=TIER_MARKERS[tier],
            color="none",
            markerfacecolor="#202020",
            markeredgecolor="#202020",
            markersize=7,
            label=TIER_LABELS[tier],
        )
        for tier in TIER_ORDER
    ]
    handles.extend(
        [
            Line2D([0], [0], color=positive_color, linewidth=2.5, label="Over-represented"),
            Line2D([0], [0], color=negative_color, linewidth=2.5, label="Under-represented"),
        ]
    )
    return handles


def save_combined(rows, cutoff: float, limit: float):
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8), facecolor="white", sharex=True)
    for idx, (ax, media_type) in enumerate(zip(axes, MEDIA_TYPE_ORDER)):
        draw_forest(ax, rows, media_type, cutoff, limit, show_ylabel=(idx == 0))
    fig.suptitle("Emotion Residuals by Commenter Tier and Media Type", fontsize=18, fontweight="bold", y=0.98)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.20, wspace=0.10)
    fig.savefig(OUTPUT_COMBINED, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_single(rows, media_type: str, cutoff: float, limit: float, output_path: Path):
    fig, ax = plt.subplots(figsize=(10.2, 6.6), facecolor="white")
    draw_forest(ax, rows, media_type, cutoff, limit, show_ylabel=True)
    ax.set_title(
        f"Emotion Residuals by Commenter Tier: {MEDIA_TYPE_LABELS[media_type]}",
        fontsize=16,
        fontweight="bold",
        color="black",
        pad=22,
    )
    ax.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.23)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    configure_fonts()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.input_csv))
    cutoff = args.significance_cutoff
    limit = residual_limit(rows, cutoff)
    save_combined(rows, cutoff, limit)
    save_single(rows, "traditional", cutoff, limit, OUTPUT_TRADITIONAL)
    save_single(rows, "emerging", cutoff, limit, OUTPUT_EMERGING)
    print(f"Wrote {OUTPUT_COMBINED}")
    print(f"Wrote {OUTPUT_TRADITIONAL}")
    print(f"Wrote {OUTPUT_EMERGING}")


if __name__ == "__main__":
    main()
