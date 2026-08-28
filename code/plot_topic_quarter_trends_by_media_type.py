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
from matplotlib.ticker import PercentFormatter


INPUT_CSV = Path("topic_by_media_type/media_type_quarter_topic_summary.csv")
OUTPUT_DIR = Path("topic_by_media_type")
OUTPUT_PNG = OUTPUT_DIR / "quarter_topic_trends_by_media_type.png"
OUTPUT_JSON = OUTPUT_DIR / "quarter_topic_trends_by_media_type_summary.json"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quarterly topic-share trends by media type.")
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


def load_rows(path: Path, tier: str):
    data = {}
    quarters = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["tier"] != tier:
                continue
            quarter = row["quarter"]
            media_type = row["media_type"]
            topic = row["topic"]
            quarters.add(quarter)
            data[(media_type, topic, quarter)] = float(row["topic_comment_share"])
    quarters = sorted(quarters)
    return data, quarters


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / OUTPUT_PNG.name
    output_json = output_dir / OUTPUT_JSON.name

    data, quarters = load_rows(Path(args.input_csv), args.tier)
    x_positions = list(range(len(quarters)))

    ncols = 2
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 11.4), facecolor="white", sharex=True)
    axes = axes.flatten()

    for ax, topic in zip(axes, TOPIC_LABELS_EN):
        for media_type in MEDIA_TYPE_ORDER:
            values = [data.get((media_type, topic, quarter), 0.0) for quarter in quarters]
            ax.plot(
                x_positions,
                values,
                marker="o",
                markersize=4.3,
                linewidth=2.2,
                color=MEDIA_TYPE_COLORS[media_type],
                label=MEDIA_TYPE_LABELS[media_type],
            )
        ax.set_title(TOPIC_LABELS_EN[topic], fontsize=11.8, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", color="#E6E1D8", linewidth=0.8)
        ax.grid(axis="x", color="#F2EFE8", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D5CFC4")
        ax.spines["bottom"].set_color("#D5CFC4")
        ax.tick_params(axis="y", labelsize=9, colors="#333333")
        ax.set_ylim(bottom=0)

    for ax in axes[-ncols:]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(quarters, rotation=35, ha="right", fontsize=9, color="#333333")
    for ax in axes[:-ncols]:
        ax.tick_params(axis="x", length=0, labelbottom=False)

    handles = [
        Line2D(
            [0],
            [0],
            color=MEDIA_TYPE_COLORS[media_type],
            marker="o",
            linewidth=2.4,
            markersize=4.5,
            label=MEDIA_TYPE_LABELS[media_type],
        )
        for media_type in MEDIA_TYPE_ORDER
    ]
    fig.suptitle("Quarterly Topic Shares by Media Type", fontsize=18, fontweight="bold", y=0.99)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, ncol=2, fontsize=10.5)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.955], h_pad=1.5, w_pad=1.6)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    output_json.write_text(
        json.dumps(
            {
                "generated_by": Path(__file__).name,
                "input_csv": str(args.input_csv),
                "tier": args.tier,
                "quarters": quarters,
                "output_png": str(output_png),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
