#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_emotion_by_media_type import EMOTION_ORDER
from analyze_topic_by_media_type import TOPICS
from channel_media_types import display_label_for_channel


TOPIC_INPUT_CSV = Path("topic_by_media_type/channel_topic_summary.csv")
EMOTION_INPUT_CSV = Path("emotion_by_media_type/channel_emotion_summary.csv")
TOPIC_SUPPORT_CSV = Path("topic_by_media_type/channel_topic_media_type_support.csv")
EMOTION_SUPPORT_CSV = Path("emotion_by_media_type/channel_emotion_media_type_support.csv")
TOPIC_OUTPUT_DIR = Path("topic_by_media_type")
EMOTION_OUTPUT_DIR = Path("emotion_by_media_type")

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
MEDIA_TYPE_COLORS = {
    "traditional": "#4F83B9",
    "emerging": "#D95D5D",
}
MEDIA_TYPE_LABELS = {
    "traditional": "Traditional",
    "emerging": "Digital / emerging",
}

TOPIC_LABELS_EN = {
    "政治行為與事件": "Political actions\nand events",
    "政黨與政治陣營": "Parties and\npolitical camps",
    "政治制度與價值": "Civic, institutional,\nand normative concepts",
    "國家與地緣政治": "Nation and\ngeopolitics",
    "政治人物": "Political figures",
    "情緒評價與社群語言": "Evaluative and community\nexpressions",
    "政治職位與機構": "Political offices and\ngovernmental bodies",
    "政策與治理議題": "Policy and\ngovernance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot channel-level topic and emotion analyses.")
    parser.add_argument("--topic-input-csv", default=TOPIC_INPUT_CSV)
    parser.add_argument("--emotion-input-csv", default=EMOTION_INPUT_CSV)
    parser.add_argument("--topic-support-csv", default=TOPIC_SUPPORT_CSV)
    parser.add_argument("--emotion-support-csv", default=EMOTION_SUPPORT_CSV)
    parser.add_argument("--topic-output-dir", default=TOPIC_OUTPUT_DIR)
    parser.add_argument("--emotion-output-dir", default=EMOTION_OUTPUT_DIR)
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
    plt.rcParams["figure.dpi"] = 120


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def channel_order(rows: list[dict]) -> list[str]:
    channel_meta = {}
    for row in rows:
        channel = row["channel"]
        meta = channel_meta.setdefault(
            channel,
            {
                "media_type": row["media_type"],
                "comment_count": int(row["comment_count"]),
            },
        )
        meta["comment_count"] = max(meta["comment_count"], int(row["comment_count"]))
    return [
        channel
        for channel, _ in sorted(
            channel_meta.items(),
            key=lambda item: (
                MEDIA_TYPE_ORDER.index(item[1]["media_type"]),
                -item[1]["comment_count"],
                item[0],
            ),
        )
    ]


def clipped_matrix(values: list[list[float]], percentile: float = 0.98) -> tuple[list[list[float]], float]:
    flat = sorted(value for row in values for value in row)
    if not flat:
        return values, 1.0
    index = min(len(flat) - 1, max(0, int(len(flat) * percentile)))
    vmax = flat[index] or max(flat) or 1.0
    return [[min(value, vmax) for value in row] for row in values], vmax


def draw_heatmap(
    rows: list[dict],
    output_png: Path,
    output_json: Path,
    *,
    item_key: str,
    item_order: list[str],
    item_labels: dict[str, str],
    metric: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    clip: bool,
) -> None:
    channels = channel_order(rows)
    lookup = {(row["channel"], row[item_key]): float(row[metric]) for row in rows}
    media_lookup = {row["channel"]: row["media_type"] for row in rows}
    matrix = [[lookup.get((channel, item), 0.0) for item in item_order] for channel in channels]
    plot_matrix, vmax = clipped_matrix(matrix) if clip else (matrix, max((v for row in matrix for v in row), default=1.0))

    height = max(7.2, 0.38 * len(channels) + 2.2)
    width = max(10.8, 1.25 * len(item_order) + 4.4)
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    image = ax.imshow(plot_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax or 1.0)

    ax.set_xticks(range(len(item_order)))
    ax.set_xticklabels([item_labels.get(item, item) for item in item_order], fontsize=9.2)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([display_label_for_channel(channel) for channel in channels], fontsize=9.0)
    for tick, channel in zip(ax.get_yticklabels(), channels):
        tick.set_color(MEDIA_TYPE_COLORS[media_lookup[channel]])
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for x in range(len(item_order) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1.0)
    for y in range(len(channels) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.030, pad=0.018)
    cbar.set_label(colorbar_label, fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.8)
    legend_handles = [
        Patch(facecolor=MEDIA_TYPE_COLORS[media_type], label=MEDIA_TYPE_LABELS[media_type])
        for media_type in MEDIA_TYPE_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, -0.07), ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "generated_by": Path(__file__).name,
        "input_metric": metric,
        "output_png": str(output_png),
        "rows": len(channels),
        "channel_labels": {channel: display_label_for_channel(channel) for channel in channels},
        "columns": item_order,
        "colorbar_clipped": clip,
        "colorbar_vmax": vmax,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def support_order(support_rows: list[dict], item_key: str) -> list[str]:
    return [
        row[item_key]
        for row in sorted(
            support_rows,
            key=lambda row: float(row["difference_channel_mean_per_1000_views"]),
        )
    ]


def draw_support_dotplot(
    rows: list[dict],
    support_rows: list[dict],
    output_png: Path,
    output_json: Path,
    *,
    item_key: str,
    item_labels: dict[str, str],
    metric: str,
    title: str,
    xlabel: str,
) -> None:
    items = support_order(support_rows, item_key)
    row_lookup = defaultdict(list)
    for row in rows:
        row_lookup[row[item_key]].append(row)

    fig, ax = plt.subplots(figsize=(11.8, max(6.6, len(items) * 0.62 + 2.0)), facecolor="white")
    y_positions = {item: index for index, item in enumerate(items)}
    offsets = {"traditional": 0.14, "emerging": -0.14}

    for item in items:
        y = y_positions[item]
        for media_type in MEDIA_TYPE_ORDER:
            values = [
                float(row[metric])
                for row in row_lookup[item]
                if row["media_type"] == media_type
            ]
            if not values:
                continue
            ax.scatter(
                values,
                [y + offsets[media_type]] * len(values),
                s=38,
                color=MEDIA_TYPE_COLORS[media_type],
                alpha=0.42,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            mean_value = sum(values) / len(values)
            ax.scatter(
                mean_value,
                y + offsets[media_type],
                marker="D",
                s=76,
                color=MEDIA_TYPE_COLORS[media_type],
                edgecolor="#222222",
                linewidth=0.55,
                zorder=4,
            )
        ax.axhline(y, color="#EFE9DF", linewidth=0.9, zorder=0)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([item_labels.get(item, item) for item in items], fontsize=9.6)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5DED4", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9C4B8")
    ax.spines["bottom"].set_color("#C9C4B8")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9.3)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MEDIA_TYPE_COLORS[media_type],
            markeredgecolor="white",
            markersize=7.5,
            label=f"{MEDIA_TYPE_LABELS[media_type]} channels",
        )
        for media_type in MEDIA_TYPE_ORDER
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="#222222",
                markersize=7.5,
                label="Channel mean",
            )
        ]
    )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "generated_by": Path(__file__).name,
        "metric": metric,
        "output_png": str(output_png),
        "item_order": items,
        "support_rows": support_rows,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def main() -> None:
    args = parse_args()
    configure_style()
    topic_output_dir = Path(args.topic_output_dir)
    emotion_output_dir = Path(args.emotion_output_dir)
    topic_output_dir.mkdir(parents=True, exist_ok=True)
    emotion_output_dir.mkdir(parents=True, exist_ok=True)

    topic_rows = read_rows(Path(args.topic_input_csv))
    emotion_rows = read_rows(Path(args.emotion_input_csv))
    topic_support_rows = read_rows(Path(args.topic_support_csv))
    emotion_support_rows = read_rows(Path(args.emotion_support_csv))

    topic_order = list(TOPICS.keys())
    emotion_order = EMOTION_ORDER

    draw_heatmap(
        topic_rows,
        topic_output_dir / "channel_topic_share_heatmap.png",
        topic_output_dir / "channel_topic_share_heatmap_summary.json",
        item_key="topic",
        item_order=topic_order,
        item_labels=TOPIC_LABELS_EN,
        metric="topic_comment_share",
        title="Channel-level lexical-domain presence shares",
        colorbar_label="Share of analyzed comments",
        cmap="YlGnBu",
        clip=False,
    )
    draw_heatmap(
        emotion_rows,
        emotion_output_dir / "channel_emotion_share_heatmap.png",
        emotion_output_dir / "channel_emotion_share_heatmap_summary.json",
        item_key="emotion",
        item_order=emotion_order,
        item_labels={emotion: emotion for emotion in emotion_order},
        metric="emotion_share",
        title="Channel-Level Distribution of Emotional Tone",
        colorbar_label="Share of analyzed comments",
        cmap="YlOrRd",
        clip=False,
    )
    draw_heatmap(
        topic_rows,
        topic_output_dir / "channel_topic_intensity_per_1000_views_heatmap.png",
        topic_output_dir / "channel_topic_intensity_per_1000_views_heatmap_summary.json",
        item_key="topic",
        item_order=topic_order,
        item_labels=TOPIC_LABELS_EN,
        metric="topic_comments_per_1000_views",
        title="View-Normalized Lexical-Domain Commenting Intensity by Channel",
        colorbar_label="Lexical-domain comments per 1,000 views",
        cmap="PuBuGn",
        clip=True,
    )
    draw_heatmap(
        emotion_rows,
        emotion_output_dir / "channel_emotion_intensity_per_1000_views_heatmap.png",
        emotion_output_dir / "channel_emotion_intensity_per_1000_views_heatmap_summary.json",
        item_key="emotion",
        item_order=emotion_order,
        item_labels={emotion: emotion for emotion in emotion_order},
        metric="emotion_per_1000_views",
        title="View-Normalized Emotional Response Intensity by Channel",
        colorbar_label="Emotion comments per 1,000 views",
        cmap="OrRd",
        clip=True,
    )
    draw_support_dotplot(
        topic_rows,
        topic_support_rows,
        topic_output_dir / "channel_topic_media_type_support_dotplot.png",
        topic_output_dir / "channel_topic_media_type_support_dotplot_summary.json",
        item_key="topic",
        item_labels=TOPIC_LABELS_EN,
        metric="topic_comments_per_1000_views",
        title="Channel-Level Lexical-Domain Intensity",
        xlabel="Lexical-domain comments per 1,000 views",
    )
    draw_support_dotplot(
        emotion_rows,
        emotion_support_rows,
        emotion_output_dir / "channel_emotion_media_type_support_dotplot.png",
        emotion_output_dir / "channel_emotion_media_type_support_dotplot_summary.json",
        item_key="emotion",
        item_labels={emotion: emotion for emotion in emotion_order},
        metric="emotion_per_1000_views",
        title="Channel-Level Heterogeneity in Emotional Response Intensity",
        xlabel="Emotion comments per 1,000 views",
    )


if __name__ == "__main__":
    main()
