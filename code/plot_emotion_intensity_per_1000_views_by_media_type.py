#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT_CSV = Path("emotion_by_media_type/media_type_emotion_summary.csv")
OUTPUT_DIR = Path("emotion_by_media_type")
OUTPUT_PNG = OUTPUT_DIR / "emotion_intensity_per_1000_views_by_media_type.png"
OUTPUT_JSON = OUTPUT_DIR / "emotion_intensity_per_1000_views_by_media_type_summary.json"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
MEDIA_TYPE_LABELS = {
    "traditional": "Traditional Media",
    "emerging": "Digital / Emerging Media",
}
MEDIA_TYPE_COLORS = {
    "traditional": "#4F83B9",
    "emerging": "#D95D5D",
}

EMOTION_ORDER = ["Neutral", "Concerned", "Happy", "Angry", "Sad", "Questioning", "Surprised", "Disgusted"]

METRICS = [
    ("per_1000_views", "Emotion comments per 1,000 views", "View-normalized intensity"),
    (
        "per_1000_video_comments",
        "Emotion comments per 1,000 YouTube comments",
        "Comment-normalized intensity",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot emotion-comment intensity per 1,000 views/comments by media type."
    )
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--tier", default="overall")
    return parser.parse_args()


def configure_style() -> None:
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


def load_rows(path: Path, tier: str) -> tuple[dict, dict, list[str]]:
    values: dict[tuple[str, str, str], float] = {}
    metadata: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            media_type = row["media_type"]
            if row["tier"] != tier or media_type not in MEDIA_TYPE_ORDER:
                continue
            metadata[media_type] = {
                "comment_count": int(row["comment_count"]),
                "unique_user_count": int(row["unique_user_count"]),
                "video_count": int(row["video_count"]),
                "video_view_count": int(row["video_view_count"]),
                "video_comment_count": int(row["video_comment_count"]),
            }
            for emotion in EMOTION_ORDER:
                values[(media_type, emotion, "per_1000_views")] = float(row[f"{emotion}_per_1000_views"])
                values[(media_type, emotion, "per_1000_video_comments")] = float(
                    row[f"{emotion}_per_1000_video_comments"]
                )

    emotion_order = sorted(
        EMOTION_ORDER,
        key=lambda emotion: max(
            values.get((media_type, emotion, "per_1000_views"), 0)
            for media_type in MEDIA_TYPE_ORDER
        ),
        reverse=True,
    )
    return values, metadata, emotion_order


def draw_metric(ax, values: dict, emotions: list[str], metric: str, xlabel: str, subtitle: str) -> None:
    y_positions = list(range(len(emotions)))
    for y, emotion in zip(y_positions, emotions):
        pair_values = [values.get((media_type, emotion, metric), 0) for media_type in MEDIA_TYPE_ORDER]
        ax.plot(pair_values, [y, y], color="#D9D3C8", linewidth=1.8, zorder=1)
        for media_type, value in zip(MEDIA_TYPE_ORDER, pair_values):
            ax.scatter(
                value,
                y,
                s=74,
                color=MEDIA_TYPE_COLORS[media_type],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(emotions, fontsize=10.2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_title(subtitle, fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#E9E5DD", linewidth=0.9)
    ax.grid(axis="y", color="#F4F1EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9C4B8")
    ax.spines["bottom"].set_color("#C9C4B8")
    ax.tick_params(axis="x", labelsize=9.5, colors="#222222")
    ax.tick_params(axis="y", length=0, colors="#111111")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / OUTPUT_PNG.name
    output_json = output_dir / OUTPUT_JSON.name

    values, metadata, emotions = load_rows(Path(args.input_csv), args.tier)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.9), facecolor="white", sharey=True)
    for ax, (metric, xlabel, subtitle) in zip(axes, METRICS):
        draw_metric(ax, values, emotions, metric, xlabel, subtitle)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MEDIA_TYPE_COLORS[media_type],
            markeredgecolor="white",
            markersize=8.5,
            label=f"{MEDIA_TYPE_LABELS[media_type]} (n={metadata[media_type]['comment_count']:,})",
        )
        for media_type in MEDIA_TYPE_ORDER
    ]
    fig.suptitle("Emotion Engagement Intensity by Media Type", fontsize=18, fontweight="bold", y=0.98)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.14, wspace=0.12)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "generated_by": Path(__file__).name,
        "input_csv": str(args.input_csv),
        "tier": args.tier,
        "output_png": str(output_png),
        "metrics": [metric for metric, _, _ in METRICS],
        "metadata": metadata,
        "emotion_order": emotions,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
