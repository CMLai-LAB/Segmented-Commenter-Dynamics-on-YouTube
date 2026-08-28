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


INPUT_CSV = Path("topic_by_media_type/traditional_vs_emerging_topic_comparison.csv")
OUTPUT_DIR = Path("topic_by_media_type")
OUTPUT_PNG = OUTPUT_DIR / "topic_intensity_difference_by_media_type.png"
OUTPUT_JSON = OUTPUT_DIR / "topic_intensity_difference_by_media_type_summary.json"

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

METRICS = [
    (
        "difference_topic_comments_per_1000_views",
        "Difference in lexical-domain comments per 1,000 views",
        "View-normalized difference",
    ),
    (
        "difference_topic_comments_per_1000_video_comments",
        "Difference in lexical-domain comments per 1,000 YouTube comments",
        "Comment-normalized difference",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot traditional-minus-emerging topic intensity differences."
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


def load_rows(path: Path, tier: str) -> tuple[list[dict], list[str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["tier"] == tier:
                rows.append(row)
    rows.sort(key=lambda row: abs(float(row["difference_topic_comments_per_1000_views"])), reverse=True)
    topics = [row["topic"] for row in rows]
    return rows, topics


def x_limit(rows: list[dict], metric: str) -> float:
    max_abs = max((abs(float(row[metric])) for row in rows), default=0.0)
    return max_abs * 1.18 if max_abs else 1.0


def draw_metric(ax, rows: list[dict], topics: list[str], metric: str, xlabel: str, subtitle: str) -> None:
    y_positions = list(range(len(topics)))
    values = [float(row[metric]) for row in rows]
    limit = x_limit(rows, metric)

    ax.axvline(0, color="#8F8A82", linewidth=1.1, zorder=1)
    ax.axvspan(0, limit, color="#EEF4FA", alpha=0.75, zorder=0)
    ax.axvspan(-limit, 0, color="#FAEEEE", alpha=0.75, zorder=0)

    for y, value in zip(y_positions, values):
        color = "#4F83B9" if value > 0 else "#D95D5D"
        ax.plot([0, value], [y, y], color=color, linewidth=2.2, alpha=0.85, zorder=2)
        ax.scatter(value, y, s=70, color=color, edgecolor="white", linewidth=0.9, zorder=3)

    ax.text(
        0.02,
        1.02,
        "Emerging higher",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#B84A4A",
        fontweight="bold",
    )
    ax.text(
        0.98,
        1.02,
        "Traditional higher",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color="#37658E",
        fontweight="bold",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TOPIC_LABELS_EN[topic] for topic in topics], fontsize=9.6)
    ax.invert_yaxis()
    ax.set_xlim(-limit, limit)
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

    rows, topics = load_rows(Path(args.input_csv), args.tier)

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.2), facecolor="white", sharey=True)
    for ax, (metric, xlabel, subtitle) in zip(axes, METRICS):
        draw_metric(ax, rows, topics, metric, xlabel, subtitle)

    fig.suptitle(
        "Traditional Minus Emerging Lexical-Domain Intensity",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.12)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "generated_by": Path(__file__).name,
        "input_csv": str(args.input_csv),
        "tier": args.tier,
        "output_png": str(output_png),
        "metrics": [metric for metric, _, _ in METRICS],
        "topic_order": topics,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
