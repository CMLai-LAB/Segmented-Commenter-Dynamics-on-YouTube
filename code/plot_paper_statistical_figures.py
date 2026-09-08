#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_topic_by_media_type import load_json_data
from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "paper_statistical_figures"

TOPIC_RESIDUAL_CSV = BASE_DIR / "topic_stat_tests_by_media_type" / "standardized_residuals.csv"
EMOTION_RESIDUAL_CSV = BASE_DIR / "emotion_stat_tests_media_type_by_tier" / "standardized_residuals.csv"
MEDIA_CI_CSV = BASE_DIR / "user_mobility_bootstrap_ci" / "media_type_mobility_bootstrap_ci.csv"
CHANNEL_CI_CSV = BASE_DIR / "user_mobility_bootstrap_ci" / "channel_mobility_bootstrap_ci.csv"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
TOPIC_SUMMARY_CSV = BASE_DIR / "topic_by_media_type" / "media_type_topic_summary.csv"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
MEDIA_TYPE_SHORT = {
    "traditional": "Traditional",
    "emerging": "Emerging",
}
MEDIA_TYPE_COLORS = {
    "traditional": "#4F83B9",
    "emerging": "#D95D5D",
}
TIER_ORDER = ["overall", "tier_2_5", "tier_6_20", "tier_21_50", "tier_51_plus"]
TIER_LABELS = {
    "overall": "Overall",
    "tier_2_5": "2-5",
    "tier_6_20": "6-20",
    "tier_21_50": "21-50",
    "tier_51_plus": "51+",
}
TOPIC_LABELS = {
    "政治行為與事件": "Political actions\nand events",
    "政黨與政治陣營": "Parties and\ncamps",
    "政治制度與價值": "Institutions\nand values",
    "國家與地緣政治": "Nation and\ngeopolitics",
    "政治人物": "Political\nfigures",
    "情緒評價與社群語言": "Evaluation and\ncommunity language",
    "政治職位與機構": "Offices and\ninstitutions",
    "政策與治理議題": "Policy and\ngovernance",
}
EMOTION_LABELS = {
    "Neutral": "Neutral",
    "Concerned": "Concerned",
    "Happy": "Happy",
    "Angry": "Angry",
    "Sad": "Sad",
    "Questioning": "Questioning",
    "Surprised": "Surprised",
    "Disgusted": "Disgusted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-ready statistical result figures.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--topic-residual-csv", default=TOPIC_RESIDUAL_CSV)
    parser.add_argument("--emotion-residual-csv", default=EMOTION_RESIDUAL_CSV)
    parser.add_argument("--media-ci-csv", default=MEDIA_CI_CSV)
    parser.add_argument("--channel-ci-csv", default=CHANNEL_CI_CSV)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--topic-summary-csv", default=TOPIC_SUMMARY_CSV)
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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_count(value) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def save_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def residual_matrix(rows: list[dict], item_column: str, item_order: list[str]) -> tuple[list[str], np.ndarray]:
    row_order = [(tier, media_type) for tier in TIER_ORDER for media_type in MEDIA_TYPE_ORDER]
    lookup = {
        (row["tier"], row["media_type"], row[item_column]): as_float(row["standardized_residual"])
        for row in rows
    }
    matrix = np.array(
        [[lookup.get((tier, media_type, item), 0.0) for item in item_order] for tier, media_type in row_order],
        dtype=float,
    )
    row_labels = [
        f"{TIER_LABELS[tier]} / {MEDIA_TYPE_SHORT[media_type]}"
        for tier, media_type in row_order
    ]
    return row_labels, matrix


def draw_residual_heatmap(
    *,
    rows: list[dict],
    item_column: str,
    item_labels: dict[str, str],
    output_png: Path,
    output_json: Path,
    title: str,
    subtitle: str,
) -> None:
    item_order = list(dict.fromkeys(row[item_column] for row in rows))
    row_labels, matrix = residual_matrix(rows, item_column, item_order)
    residual_significance = {
        (row["tier"], row["media_type"], row[item_column]): row.get("residual_reject_bh_fdr_0_05", "").lower() == "true"
        for row in rows
    }
    has_fdr_flags = any(row.get("residual_reject_bh_fdr_0_05", "") for row in rows)
    vmax = max(2.0, float(np.nanmax(np.abs(matrix))))
    vmax = min(vmax, 45.0)

    fig, ax = plt.subplots(figsize=(14.0, 7.4), facecolor="white")
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(item_order)))
    ax.set_xticklabels([item_labels.get(item, item) for item in item_order], fontsize=9.3)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.4)
    for tick, (_, media_type) in zip(ax.get_yticklabels(), [(tier, mt) for tier in TIER_ORDER for mt in MEDIA_TYPE_ORDER]):
        tick.set_color(MEDIA_TYPE_COLORS[media_type])

    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color="#555555",
    )

    for x in range(len(item_order) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1.0)
    for y in range(len(row_labels) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.8)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            tier, media_type = [(tier, mt) for tier in TIER_ORDER for mt in MEDIA_TYPE_ORDER][y]
            is_fdr_significant = residual_significance.get((tier, media_type, item_order[x]), False)
            if abs(value) >= 3 and (not has_fdr_flags or is_fdr_significant):
                text_color = "white" if abs(value) > vmax * 0.55 else "#222222"
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7.6, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
    cbar.set_label("Standardized residual", fontsize=9.6)
    cbar.ax.tick_params(labelsize=8.8)
    ax.legend(
        handles=[
            Patch(facecolor=MEDIA_TYPE_COLORS["traditional"], label="Traditional media"),
            Patch(facecolor=MEDIA_TYPE_COLORS["emerging"], label="Digital / emerging media"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.11),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_png}")

    save_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "output_png": str(output_png),
            "title": title,
            "rows": row_labels,
            "columns": item_order,
            "note": "Values are adjusted standardized residuals from media-type by category chi-square tests within each tier. When available, labels are limited to cells significant after Benjamini-Hochberg FDR correction; the visual threshold is |residual| >= 3.",
        },
    )


def draw_emotion_residual_bar_panels(
    *,
    rows: list[dict],
    output_png: Path,
    output_json: Path,
) -> None:
    """Plot tier-specific emotion residuals as diverging bars in one media direction."""
    tier_order = TIER_ORDER[1:]
    emotion_order = list(EMOTION_LABELS)
    lookup = {
        (row["tier"], row["media_type"], row["emotion"]): row
        for row in rows
    }
    values = [
        as_float(lookup[(tier, "traditional", emotion)]["standardized_residual"])
        for tier in tier_order
        for emotion in emotion_order
        if (tier, "traditional", emotion) in lookup
    ]
    limit = max(5.0, float(np.ceil(max(abs(value) for value in values) / 5.0) * 5.0))
    y_positions = np.arange(len(emotion_order))

    fig, axes = plt.subplots(2, 2, figsize=(12.1, 8.1), sharex=True, sharey=True, facecolor="white")
    for axis, tier in zip(axes.flat, tier_order):
        axis.axvspan(-3, 3, color="#F3F4F6", zorder=0)
        axis.axvline(0, color="#374151", linewidth=0.9, zorder=1)
        axis.axvline(-3, color="#9CA3AF", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
        axis.axvline(3, color="#9CA3AF", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)

        for y, emotion in enumerate(emotion_order):
            row = lookup[(tier, "traditional", emotion)]
            value = as_float(row["standardized_residual"])
            fdr_significant = row.get("residual_reject_bh_fdr_0_05", "").lower() == "true"
            displayed = fdr_significant and abs(value) >= 3
            color = MEDIA_TYPE_COLORS["traditional"] if value > 0 else MEDIA_TYPE_COLORS["emerging"]
            axis.barh(
                y,
                value,
                height=0.56,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95 if displayed else 0.35,
                zorder=2,
            )
            if displayed:
                horizontal_alignment = "left" if value >= 0 else "right"
                x_offset = 0.75 if value >= 0 else -0.75
                axis.text(
                    value + x_offset,
                    y,
                    f"{value:.1f}",
                    ha=horizontal_alignment,
                    va="center",
                    fontsize=7.8,
                    color="#1F2937",
                    zorder=4,
                )

        axis.set_title(f"{TIER_LABELS[tier]} comments", fontsize=10.8, fontweight="bold", loc="center", pad=7)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9CA3AF")
        axis.tick_params(axis="both", labelsize=8.7, colors="#374151")
        axis.set_xlim(-limit, limit)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(emotion_order, fontsize=8.8)

    axes.flat[0].set_ylim(len(emotion_order) - 0.5, -0.5)

    for axis in axes[-1, :]:
        axis.set_xlabel("Adjusted standardized residual (traditional-media direction)", fontsize=9.3)
    fig.suptitle("Emotion-label residuals by participation tier", fontsize=15.5, fontweight="bold", y=0.985)
    fig.legend(
        handles=[
            Patch(facecolor=MEDIA_TYPE_COLORS["traditional"], label="Traditional over-represented"),
            Patch(facecolor=MEDIA_TYPE_COLORS["emerging"], label="Emerging over-represented"),
            Line2D([0], [0], color="#9CA3AF", linestyle=(0, (2, 2)), linewidth=1.0, label="|residual| = 3"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.13, hspace=0.34, wspace=0.18)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_png}")

    save_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "output_png": str(output_png),
            "tiers": tier_order,
            "emotions": emotion_order,
            "direction": "Adjusted standardized residuals are plotted once in the traditional-media direction; positive bars indicate traditional-media over-representation and negative bars indicate emerging-media over-representation.",
            "display_rule": "Numeric labels are shown only for FDR-significant values with absolute residual at least three.",
        },
    )


def metric_rows(rows: list[dict], metric_order: list[str]) -> list[dict]:
    lookup = {row["metric"]: row for row in rows}
    return [lookup[metric] for metric in metric_order if metric in lookup]


def draw_ci_plot(
    *,
    rows: list[dict],
    metric_order: list[str],
    labels: dict[str, str],
    output_png: Path,
    output_json: Path,
    title: str,
    subtitle: str,
    show_value_labels: bool = True,
) -> None:
    selected = metric_rows(rows, metric_order)
    y = np.arange(len(selected))
    observed = np.array([as_float(row["observed_value"]) for row in selected])
    lower = np.array([as_float(row["ci_lower"]) for row in selected])
    upper = np.array([as_float(row["ci_upper"]) for row in selected])
    xerr = np.vstack([observed - lower, upper - observed])

    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.58 * len(selected) + 2.2)), facecolor="white")
    ax.errorbar(
        observed,
        y,
        xerr=xerr,
        fmt="o",
        color="#243B53",
        ecolor="#7D8FA3",
        elinewidth=2.2,
        capsize=4,
        markersize=7.5,
        markerfacecolor="#243B53",
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(row["metric"], row["metric"]) for row in selected], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.0, max(upper.max() + 0.08, 0.18)) if len(upper) else 1.0)
    ax.set_xlabel("Proportion with 95% descriptive percentile interval", fontsize=10.5)
    ax.set_title(title, fontsize=16.5, fontweight="bold", y=1.055 if not subtitle else 1.095, pad=12)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0, color="#555555")
    ax.grid(axis="x", color="#e8e8e8", linewidth=1.0)
    ax.grid(axis="y", color="#f3f3f3", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d4d4d4")
    if show_value_labels:
        for yi, value in zip(y, observed):
            ax.text(value + 0.012, yi, f"{value:.3f}", va="center", ha="left", fontsize=8.8, color="#333333")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_png}")

    save_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "output_png": str(output_png),
            "title": title,
            "show_value_labels": show_value_labels,
            "metrics": [
                {
                    "metric": row["metric"],
                    "label": labels.get(row["metric"], row["metric"]),
                    "observed_value": row["observed_value"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "n_units": row["n_units"],
                    "n_bootstrap": row["n_bootstrap"],
                }
                for row in selected
            ],
        },
    )


def draw_media_mobility_ci(rows: list[dict], output_dir: Path) -> None:
    draw_ci_plot(
        rows=rows,
        metric_order=[
            "jaccard_overlap",
            "overlap_share_of_traditional",
            "overlap_share_of_emerging",
            "state_change_share",
            "side_switch_share",
            "single_to_both_share",
            "both_to_single_share",
            "retained_share_of_union",
        ],
        labels={
            "jaccard_overlap": "Pooled monthly T/E overlap (Jaccard)",
            "overlap_share_of_traditional": "Overlap share of traditional users",
            "overlap_share_of_emerging": "Overlap share of emerging users",
            "state_change_share": "State change among continuing users",
            "side_switch_share": "Direct traditional-emerging switch",
            "single_to_both_share": "Single-media to both media types",
            "both_to_single_share": "Both media types to single-media",
            "retained_share_of_union": "Retained users among monthly union",
        },
        output_png=output_dir / "media_type_mobility_bootstrap_ci.png",
        output_json=output_dir / "media_type_mobility_bootstrap_ci_summary.json",
        title="Descriptive Bootstrap Intervals for Media-Type Mobility",
        subtitle="Commenters with at least two in-corpus comments over the full observation window",
        show_value_labels=False,
    )

def load_overall_topic_summary(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    summary = {}
    for row in rows:
        if row["tier"] != "overall":
            continue
        media_type = row["media_type"]
        item = summary.setdefault(media_type, {})
        item["analyzed_comments"] = max(item.get("analyzed_comments", 0), parse_count(row["comment_count"]))
        item["unique_users"] = max(item.get("unique_users", 0), parse_count(row["unique_user_count"]))
    return summary


def draw_data_overview(videos_json: Path, topic_summary_csv: Path, output_png: Path, output_json: Path) -> None:
    video_counts = Counter()
    view_counts = Counter()
    video_comment_counts = Counter()
    for video in load_json_data(videos_json):
        video_id = video.get("id") or video.get("video_id")
        if not video_id:
            continue
        snippet = video.get("snippet") or {}
        statistics = video.get("statistics") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        media_type = media_type_for_channel(channel)
        if media_type not in MEDIA_TYPE_ORDER:
            continue
        video_counts[media_type] += 1
        view_counts[media_type] += parse_count(statistics.get("viewCount"))
        video_comment_counts[media_type] += parse_count(statistics.get("commentCount"))

    comment_summary = load_overall_topic_summary(topic_summary_csv)
    metrics = [
        ("Videos", video_counts),
        ("Views", view_counts),
        ("Video comments", video_comment_counts),
        ("Analyzed comments", {media_type: comment_summary.get(media_type, {}).get("analyzed_comments", 0) for media_type in MEDIA_TYPE_ORDER}),
        ("Unique commenters", {media_type: comment_summary.get(media_type, {}).get("unique_users", 0) for media_type in MEDIA_TYPE_ORDER}),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16.8, 4.6), facecolor="white")
    for ax, (metric, values) in zip(axes, metrics):
        data = [values.get(media_type, 0) for media_type in MEDIA_TYPE_ORDER]
        x = np.arange(len(MEDIA_TYPE_ORDER))
        ax.bar(
            x,
            data,
            color=[MEDIA_TYPE_COLORS[media_type] for media_type in MEDIA_TYPE_ORDER],
            width=0.62,
        )
        ax.set_title(metric, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([MEDIA_TYPE_SHORT[media_type] for media_type in MEDIA_TYPE_ORDER], fontsize=9.2)
        ax.tick_params(axis="y", labelsize=8.8)
        ax.grid(axis="y", color="#ececec", linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#d6d6d6")
        ymax = max(data) if data else 0
        for xi, value in zip(x, data):
            label = f"{value:,}"
            ax.text(xi, value + ymax * 0.025 if ymax else 0.02, label, ha="center", va="bottom", fontsize=8.2)

    fig.suptitle("Dataset Overview by Media Type", fontsize=17, fontweight="bold", y=1.03)
    legend_handles = [
        Patch(facecolor=MEDIA_TYPE_COLORS[media_type], label=MEDIA_TYPE_LABELS[media_type])
        for media_type in MEDIA_TYPE_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_png}")

    save_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "output_png": str(output_png),
            "metrics": {
                metric: {media_type: values.get(media_type, 0) for media_type in MEDIA_TYPE_ORDER}
                for metric, values in metrics
            },
        },
    )


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_rows = read_csv(Path(args.topic_residual_csv))
    emotion_rows = read_csv(Path(args.emotion_residual_csv))
    media_ci_rows = read_csv(Path(args.media_ci_csv))
    channel_ci_rows = read_csv(Path(args.channel_ci_csv))

    draw_data_overview(
        Path(args.videos_json),
        Path(args.topic_summary_csv),
        output_dir / "dataset_overview_by_media_type.png",
        output_dir / "dataset_overview_by_media_type_summary.json",
    )
    draw_residual_heatmap(
        rows=topic_rows,
        item_column="topic",
        item_labels=TOPIC_LABELS,
        output_png=output_dir / "topic_standardized_residual_heatmap.png",
        output_json=output_dir / "topic_standardized_residual_heatmap_summary.json",
        title="Topic Response Residuals by Media Type and Participation Tier",
        subtitle="Positive values indicate categories observed more often than expected under independence.",
    )
    draw_emotion_residual_bar_panels(
        rows=emotion_rows,
        output_png=output_dir / "emotion_standardized_residual_diverging_bars.png",
        output_json=output_dir / "emotion_standardized_residual_diverging_bars_summary.json",
    )
    draw_media_mobility_ci(media_ci_rows, output_dir)
    draw_ci_plot(
        rows=channel_ci_rows,
        metric_order=[
            "same_channel_share",
            "same_media_different_channel_share",
            "cross_media_share",
            "changed_channel_share",
            "inactive_next_month_share_of_from_active",
            "retained_next_month_share_of_from_active",
        ],
        labels={
            "same_channel_share": "Same primary channel",
            "same_media_different_channel_share": "Different channel, same media type",
            "cross_media_share": "Different media type",
            "changed_channel_share": "Changed primary channel",
            "inactive_next_month_share_of_from_active": "Inactive in next month",
            "retained_next_month_share_of_from_active": "Retained in next month",
        },
        output_png=output_dir / "channel_mobility_bootstrap_ci.png",
        output_json=output_dir / "channel_mobility_bootstrap_ci_summary.json",
        title="Descriptive Bootstrap Intervals for Channel-Level Mobility",
        subtitle="Transition estimates are based on users active in adjacent month-pairs.",
    )


if __name__ == "__main__":
    main()
