#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create the compact main-text figures selected for the EPJ Data Science draft."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TIMELINE_CSV = BASE_DIR / "analysis_strengthening" / "event_timeline" / "monthly_activity_event_timeline_template.csv"
EVENTS_CSV = BASE_DIR / "analysis_strengthening" / "event_timeline" / "event_annotations.csv"
CORPUS_COMPARISON_JSON = BASE_DIR / "all_vs_recall_video_comparison.json"
VIDEO_MEDIA_TYPE_CSV = BASE_DIR / "analysis_strengthening" / "event_timeline" / "monthly_in_scope_video_counts_by_media_type.csv"
TOPIC_CSV = BASE_DIR / "topic_by_media_type" / "media_type_quarter_topic_summary.csv"
TOPIC_COMPARISON_CSV = BASE_DIR / "topic_by_media_type" / "traditional_vs_emerging_topic_comparison.csv"
MAIN_DIR = BASE_DIR / "paper_selected_figures" / "main"

TRADITIONAL_COLOR = "#4F83B9"
EMERGING_COLOR = "#D95D5D"
TOPIC_ORDER = [
    "政治行為與事件",
    "政黨與政治陣營",
    "政治制度與價值",
    "國家與地緣政治",
    "政治人物",
    "情緒評價與社群語言",
    "政治職位與機構",
    "政策與治理議題",
]
TOPIC_LABELS = {
    "政治行為與事件": "Political actions and events",
    "政黨與政治陣營": "Parties and political camps",
    "政治制度與價值": "Civic, institutional, and normative concepts",
    "國家與地緣政治": "Nation and geopolitics",
    "政治人物": "Political figures",
    "情緒評價與社群語言": "Evaluative and community expressions",
    "政治職位與機構": "Political offices and governmental bodies",
    "政策與治理議題": "Policy and governance",
}
TOPIC_PANEL_LABELS = {
    **TOPIC_LABELS,
    "政治制度與價值": "Civic, institutional,\nand normative concepts",
    "情緒評價與社群語言": "Evaluative and community\nexpressions",
    "政治職位與機構": "Political offices and\ngovernmental bodies",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected EPJ Data Science main-text figures.")
    parser.add_argument("--timeline-csv", default=TIMELINE_CSV)
    parser.add_argument("--events-csv", default=EVENTS_CSV)
    parser.add_argument("--corpus-comparison-json", default=CORPUS_COMPARISON_JSON)
    parser.add_argument("--video-media-type-csv", default=VIDEO_MEDIA_TYPE_CSV)
    parser.add_argument("--topic-csv", default=TOPIC_CSV)
    parser.add_argument("--topic-comparison-csv", default=TOPIC_COMPARISON_CSV)
    parser.add_argument("--main-dir", default=MAIN_DIR)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def month_position(months: list[str], event_date: str) -> float | None:
    event_month = event_date[:7]
    if event_month not in months:
        return None
    index = months.index(event_month)
    try:
        day = int(event_date[8:10])
    except (IndexError, ValueError):
        day = 15
    return index + min(max(day, 1), 28) / 31.0 - 0.5


def quarter_ticks(months: list[str]) -> tuple[list[int], list[str]]:
    ticks = []
    labels = []
    for index, month in enumerate(months):
        year, month_number = month.split("-")
        if month_number in {"01", "04", "07", "10"}:
            ticks.append(index)
            if index == len(months) - 1 and month_number == "10":
                labels.append(f"{year} Oct")
            else:
                labels.append(f"{year} Q{(int(month_number) - 1) // 3 + 1}")
    return ticks, labels


def add_event_markers(ax, months: list[str], events: list[dict[str, str]]) -> None:
    short_labels = {
        "CEC confirmation: 19 cases proceed to Stage II petitioning": (
            "CEC confirmation: 19 recall cases entered\nthe second-stage petition phase"
        ),
        "CEC-announced Stage II petition submission deadline": (
            "CEC deadline for second-stage\npetition submissions"
        ),
        "First statutory recall voting day": "First statutory recall voting day",
        "Second statutory recall voting day": "Second statutory recall voting day",
    }
    for event_index, event in enumerate(events):
        position = month_position(months, event["event_date"])
        if position is None:
            continue
        ax.axvline(position, color="#111111", linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.72, zorder=4)
        y_position = 0.97 if event_index % 2 == 0 else 0.80
        ax.annotate(
            short_labels.get(event["political_event_label"], event["political_event_label"]),
            xy=(position, 0.99),
            xycoords=("data", "axes fraction"),
            xytext=(position, y_position),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=7.7,
            color="#111111",
            arrowprops={"arrowstyle": "-", "color": "#111111", "lw": 0.7},
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#111111", "alpha": 0.96},
            zorder=5,
        )


def add_procedural_milestone_key(ax) -> None:
    ax.text(
        0.985,
        0.965,
        "Procedural milestones (2025)\n"
        "27 Feb: CEC confirmation of 19 second-stage cases\n"
        "28 Apr: CEC deadline for petition submissions\n"
        "26 Jul: First statutory recall voting day\n"
        "23 Aug: Second statutory recall voting day",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#111111",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#111111", "alpha": 0.97},
        zorder=6,
    )


def plot_activity_timeline(
    timeline_csv: Path,
    events_csv: Path,
    video_media_type_csv: Path,
    output_dir: Path,
) -> None:
    del timeline_csv  # Retained as a CLI argument for compatibility with earlier figure versions.
    events = read_csv(events_csv)
    rows = read_csv(video_media_type_csv)
    months = [row["month"] for row in rows]
    x = np.arange(len(months))
    traditional_counts = np.array([int(row["traditional_video_count"]) for row in rows])
    emerging_counts = np.array([int(row["emerging_video_count"]) for row in rows])
    total_counts = traditional_counts + emerging_counts

    fig, ax = plt.subplots(figsize=(14.0, 7.3), facecolor="white")
    ax.bar(
        x,
        traditional_counts,
        width=0.72,
        color=TRADITIONAL_COLOR,
        edgecolor="white",
        linewidth=0.5,
        label="Traditional media",
        zorder=3,
    )
    ax.bar(
        x,
        emerging_counts,
        bottom=traditional_counts,
        width=0.72,
        color=EMERGING_COLOR,
        edgecolor="white",
        linewidth=0.5,
        label="Emerging media",
        zorder=3,
    )

    for event in events:
        event_month = event["event_date"][:7]
        if event_month in months:
            position = months.index(event_month)
            ax.axvspan(position - 0.5, position + 0.5, color="#F3B6B6", alpha=0.28, zorder=0)

    maximum = total_counts.max()
    for index, total in enumerate(total_counts):
        if total > 0:
            ax.text(
                index,
                total + 0.018 * maximum,
                f"{total:,}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#7A1F1F",
                fontweight="bold",
                zorder=4,
            )

    ax.set_ylabel("Number of videos", fontsize=11.2)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=34, ha="right", fontsize=8.7, color="#111111")
    ax.set_xlim(-0.75, len(months) - 0.25)
    # Reserve a dedicated annotation band above the tallest monthly bar so the
    # procedural-milestone key cannot obscure count or percentage labels.
    ax.set_ylim(0, 1.42 * maximum)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#111111")
    ax.tick_params(axis="y", labelsize=9.0, colors="#111111")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    add_procedural_milestone_key(ax)
    fig.suptitle(
        "Monthly volume of in-scope recall-related videos by media type and procedural milestones",
        fontsize=16.5,
        fontweight="bold",
        color="#111111",
        y=0.985,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.17)

    output_png = output_dir / "00_overview" / "Fig_00_corpus_and_commenting_activity_timeline.png"
    output_json = output_png.with_name("Fig_00_corpus_and_commenting_activity_timeline_summary.json")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    output_json.write_text(
        json.dumps(
            {
                "generated_by": Path(__file__).name,
                "video_media_type_csv": str(video_media_type_csv),
                "events_csv": str(events_csv),
                "output_png": str(output_png),
                "month_range": [months[0], months[-1]],
                "event_annotations": events,
                "note": "Bars show monthly videos retained by the event-query rule, stacked by mutually exclusive media type.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def plot_topic_difference_panels(topic_csv: Path, output_dir: Path) -> None:
    rows = [row for row in read_csv(topic_csv) if row["tier"] == "overall"]
    quarters = sorted({row["quarter"] for row in rows})
    lookup = {
        (row["media_type"], row["topic"], row["quarter"]): float(row["topic_comment_share"])
        for row in rows
    }
    matrix = np.array(
        [
            [
                100
                * (
                    lookup.get(("traditional", topic, quarter), 0.0)
                    - lookup.get(("emerging", topic, quarter), 0.0)
                )
                for quarter in quarters
            ]
            for topic in TOPIC_ORDER
        ]
    )
    maximum = max(2.0, float(np.nanmax(np.abs(matrix))))
    maximum = float(np.ceil(maximum / 2.0) * 2.0)
    x = np.arange(len(quarters))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.1), sharex=True, sharey=True, facecolor="white")
    for axis, topic, values in zip(axes.flat, TOPIC_ORDER, matrix):
        axis.axhline(0, color="#6B7280", linewidth=0.85, zorder=1)
        axis.plot(x, values, color="#4B5563", linewidth=1.45, zorder=2)
        point_colors = [TRADITIONAL_COLOR if value >= 0 else EMERGING_COLOR for value in values]
        axis.scatter(x, values, s=28, color=point_colors, edgecolor="white", linewidth=0.45, zorder=3)
        axis.set_title(TOPIC_PANEL_LABELS[topic], fontsize=9.0, fontweight="bold", loc="center", pad=5)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9CA3AF")
        axis.set_ylim(-maximum, maximum)
        axis.tick_params(axis="both", labelsize=8.1, colors="#374151")

    quarter_labels = [quarter.replace("-Q", " Q") for quarter in quarters]
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels(quarter_labels, rotation=35, ha="right")
    for axis in axes[:, 0]:
        axis.set_ylabel("Traditional - emerging\n(pp)", fontsize=8.8)
    fig.suptitle("Quarterly lexical-domain differences by media type", fontsize=15.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.11, hspace=0.52, wspace=0.22)

    output_png = output_dir / "01_topic" / "Fig_03_quarterly_topic_media_type_difference_panels.png"
    output_json = output_png.with_name("Fig_03_quarterly_topic_media_type_difference_panels_summary.json")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    output_json.write_text(
        json.dumps(
            {
                "generated_by": Path(__file__).name,
                "topic_csv": str(topic_csv),
                "output_png": str(output_png),
                "tier": "overall",
                "quarters": quarters,
                "topics": TOPIC_ORDER,
                "difference_unit": "traditional lexical-domain share minus emerging lexical-domain share, percentage points",
                "quarterly_differences_pp": matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_topic_tier_difference_dotplot(comparison_csv: Path, output_dir: Path) -> None:
    tier_order = ["low_2_5", "mid_6_20", "core_21_50", "high_51_plus"]
    tier_labels = ["2--5", "6--20", "21--50", "51+"]
    rows = [row for row in read_csv(comparison_csv) if row["tier"] in tier_order]
    lookup = {
        (row["topic"], row["tier"]): 100.0 * float(row["difference_traditional_minus_emerging"])
        for row in rows
    }
    matrix = np.array(
        [[lookup[(topic, tier)] for tier in tier_order] for topic in TOPIC_ORDER]
    )
    maximum = max(2.0, float(np.nanmax(np.abs(matrix))))
    maximum = float(np.ceil(maximum / 1.0) * 1.0)
    y = np.arange(len(TOPIC_ORDER))
    tier_colors = ["#4F83B9", "#3D9B7A", "#D98C3F", "#B85450"]

    fig, ax = plt.subplots(figsize=(11.9, 6.7), facecolor="white")
    for index, (tier_label, color) in enumerate(zip(tier_labels, tier_colors)):
        values = matrix[:, index]
        ax.scatter(
            values,
            y,
            s=47,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
            label=tier_label,
        )
    for y_position, row in enumerate(matrix):
        ax.plot(
            row,
            np.full(len(row), y_position),
            color="#9CA3AF",
            linewidth=0.9,
            alpha=0.75,
            zorder=2,
        )
    ax.axvline(0, color="#374151", linewidth=0.95, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([TOPIC_LABELS[topic] for topic in TOPIC_ORDER], fontsize=9.9)
    ax.invert_yaxis()
    ax.set_xlim(-maximum, maximum)
    ax.set_xlabel("Traditional minus emerging lexical-domain share (percentage points)", fontsize=10.4, labelpad=8)
    ax.set_title("Lexical-domain differences by participation tier", fontsize=15.5, fontweight="bold", pad=16)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    ax.tick_params(axis="x", labelsize=9.0, colors="#374151")
    ax.tick_params(axis="y", length=0, colors="#111111")
    legend = ax.legend(
        title="Total in-corpus comments",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=8.8,
        title_fontsize=8.8,
    )
    legend._legend_box.align = "left"
    fig.tight_layout()

    output_png = output_dir / "01_topic" / "Fig_05_topic_tier_media_type_difference_dotplot.png"
    output_json = output_png.with_name("Fig_05_topic_tier_media_type_difference_dotplot_summary.json")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    output_json.write_text(
        json.dumps(
            {
                "generated_by": Path(__file__).name,
                "comparison_csv": str(comparison_csv),
                "output_png": str(output_png),
                "tiers": tier_order,
                "tier_labels": tier_labels,
                "topics": TOPIC_ORDER,
                "difference_unit": "traditional lexical-domain share minus emerging lexical-domain share, percentage points",
                "tier_differences_pp": matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def main() -> None:
    args = parse_args()
    configure_style()
    main_dir = Path(args.main_dir)
    plot_activity_timeline(
        Path(args.timeline_csv),
        Path(args.events_csv),
        Path(args.video_media_type_csv),
        main_dir,
    )
    plot_topic_difference_panels(Path(args.topic_csv), main_dir)
    plot_topic_tier_difference_dotplot(Path(args.topic_comparison_csv), main_dir)


if __name__ == "__main__":
    main()
