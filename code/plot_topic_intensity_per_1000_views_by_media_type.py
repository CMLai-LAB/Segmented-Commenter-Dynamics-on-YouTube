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


INPUT_CSV = Path("topic_by_media_type/media_type_topic_summary.csv")
OUTPUT_DIR = Path("topic_by_media_type")
OUTPUT_PNG = OUTPUT_DIR / "topic_intensity_per_1000_views_by_media_type.png"
OUTPUT_JSON = OUTPUT_DIR / "topic_intensity_per_1000_views_by_media_type_summary.json"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
MEDIA_TYPE_LABELS = {
    "traditional": "Traditional Media",
    "emerging": "Digital / Emerging Media",
}
MEDIA_TYPE_COLORS = {
    "traditional": "#4F83B9",
    "emerging": "#D95D5D",
}

TOPIC_LABELS_EN = {
    "政治行為與事件": "Political actions\nand events",
    "政黨與政治陣營": "Parties and\npolitical camps",
    "政治制度與價值": "Political institutions\nand values",
    "國家與地緣政治": "Nation and\ngeopolitics",
    "政治人物": "Political figures",
    "情緒評價與社群語言": "Emotion, evaluation,\nand community language",
    "政治職位與機構": "Political offices\nand institutions",
    "政策與治理議題": "Policy and\ngovernance",
}

METRICS = [
    (
        "topic_comments_per_1000_views",
        "Topic comments per 1,000 views",
        "View-normalized intensity",
    ),
    (
        "topic_comments_per_1000_video_comments",
        "Topic comments per 1,000 YouTube comments",
        "Comment-normalized intensity",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot topic-comment intensity per 1,000 views/comments by media type."
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


def load_rows(path: Path, tier: str) -> tuple[dict, dict]:
    rows: dict[tuple[str, str], dict] = {}
    metadata: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["tier"] != tier or row["media_type"] not in MEDIA_TYPE_ORDER:
                continue
            topic = row["topic"]
            media_type = row["media_type"]
            rows[(media_type, topic)] = row
            metadata[media_type] = {
                "comment_count": int(row["comment_count"]),
                "unique_user_count": int(row["unique_user_count"]),
                "video_count": int(row["video_count"]),
                "video_view_count": int(row["video_view_count"]),
                "video_comment_count": int(row["video_comment_count"]),
            }

    topics = sorted(
        TOPIC_LABELS_EN,
        key=lambda topic: max(
            float(rows.get((media_type, topic), {}).get("topic_comments_per_1000_views", 0))
            for media_type in MEDIA_TYPE_ORDER
        ),
        reverse=True,
    )
    return {"rows": rows, "topics": topics}, metadata


def draw_metric(ax, rows: dict, topics: list[str], metric: str, xlabel: str, subtitle: str) -> None:
    y_positions = list(range(len(topics)))
    for y, topic in zip(y_positions, topics):
        values = [
            float(rows.get((media_type, topic), {}).get(metric, 0))
            for media_type in MEDIA_TYPE_ORDER
        ]
        ax.plot(values, [y, y], color="#D9D3C8", linewidth=1.8, zorder=1)
        for media_type, value in zip(MEDIA_TYPE_ORDER, values):
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
    ax.set_yticklabels([TOPIC_LABELS_EN[topic] for topic in topics], fontsize=9.6)
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

    data, metadata = load_rows(Path(args.input_csv), args.tier)
    rows = data["rows"]
    topics = data["topics"]

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.2), facecolor="white", sharey=True)
    for ax, (metric, xlabel, subtitle) in zip(axes, METRICS):
        draw_metric(ax, rows, topics, metric, xlabel, subtitle)

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
    fig.suptitle(
        "Topic Engagement Intensity by Media Type",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
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
        "topic_order": topics,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
