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
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


INPUT_DIR = Path("user_mobility_by_media_type")
OUTPUT_DIR = INPUT_DIR

STATE_ORDER = ["traditional_only", "emerging_only", "both"]
CHURN_STATE_ORDER = ["inactive", "traditional_only", "emerging_only", "both"]
STATE_LABELS = {
    "inactive": "Inactive",
    "traditional_only": "Traditional only",
    "emerging_only": "Digital / emerging only",
    "both": "Both media types",
}
STATE_SHORT_LABELS = {
    "inactive": "Inactive",
    "traditional_only": "Traditional\nonly",
    "emerging_only": "Digital /\nemerging only",
    "both": "Both",
}
STATE_COLORS = {
    "inactive": "#B8B4AA",
    "traditional_only": "#4F83B9",
    "emerging_only": "#D95D5D",
    "both": "#4FA66A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot monthly user mobility between media types.")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
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


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_state_share(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "monthly_user_media_state_summary.csv")
    months = sorted({row["month"] for row in rows})
    lookup = {(row["month"], row["state"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(13.8, 6.4), facecolor="white")
    bottoms = [0.0] * len(months)
    x_positions = list(range(len(months)))

    for state in STATE_ORDER:
        values = [float(lookup[(month, state)]["user_share"]) for month in months]
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.8,
            width=0.76,
            label=STATE_LABELS[state],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Monthly Distribution of User Media-Participation States", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Share of active commenters", fontsize=10.5)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(months, rotation=38, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#E7E1D8", linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9C4B8")
    ax.spines["bottom"].set_color("#C9C4B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    output_png = output_dir / "monthly_user_media_state_share.png"
    output_json = output_dir / "monthly_user_media_state_share_summary.json"
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "input_csv": str(input_dir / "monthly_user_media_state_summary.csv"),
            "output_png": str(output_png),
            "months": months,
            "states": STATE_ORDER,
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def plot_overlap(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "monthly_user_overlap_summary.csv")
    months = [row["month"] for row in rows]
    x_positions = list(range(len(months)))
    series = [
        ("jaccard_overlap", "Jaccard overlap", "#2F6F4E"),
        ("overlap_share_of_traditional", "Overlap share of traditional users", "#4F83B9"),
        ("overlap_share_of_emerging", "Overlap share of digital / emerging users", "#D95D5D"),
    ]

    fig, ax = plt.subplots(figsize=(13.8, 6.2), facecolor="white")
    for key, label, color in series:
        values = [float(row[key]) for row in rows]
        ax.plot(x_positions, values, color=color, linewidth=2.4, marker="o", markersize=5.2, label=label)

    ax.set_title("Monthly Overlap Between Traditional and Digital/Emerging Commenters", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("User overlap rate", fontsize=10.5)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(months, rotation=38, ha="right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E7E1D8", linewidth=0.85)
    ax.grid(axis="x", color="#F2EFE8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9C4B8")
    ax.spines["bottom"].set_color("#C9C4B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23)

    output_png = output_dir / "monthly_user_overlap_rates.png"
    output_json = output_dir / "monthly_user_overlap_rates_summary.json"
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "input_csv": str(input_dir / "monthly_user_overlap_summary.csv"),
            "output_png": str(output_png),
            "series": [key for key, _, _ in series],
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def plot_mobility_index(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "monthly_user_mobility_index.csv")
    month_pairs = [row["month_pair"].replace("->", "\n") for row in rows]
    x_positions = list(range(len(month_pairs)))
    series = [
        ("state_change_share", "Changed participation state", "#7B5EA7"),
        ("side_switch_share", "Direct side switch", "#CF6A3A"),
        ("single_to_both_share", "Single-side to both", "#4FA66A"),
        ("both_to_single_share", "Both to single-side", "#A95D74"),
        ("retained_share_of_union", "Retained in next month", "#4F83B9"),
    ]

    fig, ax = plt.subplots(figsize=(14.6, 6.8), facecolor="white")
    for key, label, color in series:
        values = [float(row[key]) for row in rows]
        ax.plot(x_positions, values, color=color, linewidth=2.25, marker="o", markersize=4.7, label=label)

    ax.set_title("Monthly User Mobility Between Media-Participation States", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Share of users", fontsize=10.5)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(month_pairs, fontsize=8.4)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E7E1D8", linewidth=0.85)
    ax.grid(axis="x", color="#F2EFE8", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9C4B8")
    ax.spines["bottom"].set_color("#C9C4B8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)

    output_png = output_dir / "monthly_user_mobility_index.png"
    output_json = output_dir / "monthly_user_mobility_index_summary.json"
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "input_csv": str(input_dir / "monthly_user_mobility_index.csv"),
            "output_png": str(output_png),
            "series": [key for key, _, _ in series],
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def plot_transition_heatmap(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "monthly_user_transition_summary.csv")
    month_pairs = []
    for row in rows:
        if row["month_pair"] not in month_pairs:
            month_pairs.append(row["month_pair"])
    transitions = [f"{a}->{b}" for a in ["T", "E", "B"] for b in ["T", "E", "B"]]
    lookup = {(row["month_pair"], row["transition"]): float(row["pair_share"]) for row in rows}
    matrix = [[lookup.get((month_pair, transition), 0.0) for transition in transitions] for month_pair in month_pairs]

    fig, ax = plt.subplots(figsize=(12.8, max(7.2, len(month_pairs) * 0.35 + 2.2)), facecolor="white")
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_title("Month-to-Month Transitions Among Continuing Active Commenters", fontsize=17, fontweight="bold", pad=16)
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, fontsize=9.3)
    ax.set_yticks(range(len(month_pairs)))
    ax.set_yticklabels(month_pairs, fontsize=8.5)
    ax.set_xlabel("Transition between media-participation states", fontsize=10.3)
    ax.set_ylabel("Adjacent month pair", fontsize=10.3)
    ax.tick_params(length=0)
    for x in range(len(transitions) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.8)
    for y in range(len(month_pairs) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    cbar.set_label("Share of continuing active users", fontsize=9.5)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    note = "T = traditional only; E = digital / emerging only; B = both media types"
    fig.text(0.5, 0.025, note, ha="center", va="center", fontsize=9.6, color="#333333")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)

    output_png = output_dir / "monthly_user_transition_heatmap.png"
    output_json = output_dir / "monthly_user_transition_heatmap_summary.json"
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "input_csv": str(input_dir / "monthly_user_transition_summary.csv"),
            "output_png": str(output_png),
            "month_pairs": month_pairs,
            "transitions": transitions,
            "metric": "pair_share",
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def draw_matrix(ax, rows: list[dict], matrix_type: str, states: list[str], title: str) -> None:
    filtered = [row for row in rows if row["matrix_type"] == matrix_type]
    lookup = {(row["from_state"], row["to_state"]): row for row in filtered}
    matrix = [[float(lookup[(from_state, to_state)]["row_share"]) for to_state in states] for from_state in states]
    image = ax.imshow(matrix, aspect="equal", cmap="YlGnBu", vmin=0, vmax=max(max(row) for row in matrix) or 1)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([STATE_SHORT_LABELS[state] for state in states], fontsize=9.3)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([STATE_SHORT_LABELS[state] for state in states], fontsize=9.3)
    ax.set_xlabel("To state", fontsize=10)
    ax.set_ylabel("From state", fontsize=10)
    ax.tick_params(length=0)
    for y, from_state in enumerate(states):
        for x, to_state in enumerate(states):
            row = lookup[(from_state, to_state)]
            value = float(row["row_share"])
            count = int(row["user_count"])
            color = "white" if value > 0.45 else "#222222"
            ax.text(x, y, f"{value:.1%}\n({count:,})", ha="center", va="center", fontsize=8.5, color=color)
    for x in range(len(states) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1.1)
    for y in range(len(states) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=1.1)
    return image


def plot_overall_transition_matrix(input_dir: Path, output_dir: Path) -> None:
    rows = read_rows(input_dir / "monthly_user_transition_overall_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.2), facecolor="white")
    image_a = draw_matrix(
        axes[0],
        rows,
        "continuing_active_users",
        STATE_ORDER,
        "Continuing active users",
    )
    image_b = draw_matrix(
        axes[1],
        rows,
        "including_inactive_state",
        CHURN_STATE_ORDER,
        "Including inactive state",
    )
    fig.suptitle("Aggregated Month-to-Month User Transition Matrices", fontsize=17, fontweight="bold", y=0.99)
    cbar = fig.colorbar(image_b, ax=axes, fraction=0.025, pad=0.018)
    cbar.set_label("Row-normalized transition share", fontsize=9.5)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    fig.subplots_adjust(top=0.82, right=0.88, wspace=0.30)

    output_png = output_dir / "monthly_user_transition_overall_matrix.png"
    output_json = output_dir / "monthly_user_transition_overall_matrix_summary.json"
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    write_summary(
        output_json,
        {
            "generated_by": Path(__file__).name,
            "input_csv": str(input_dir / "monthly_user_transition_overall_matrix.csv"),
            "output_png": str(output_png),
            "matrices": ["continuing_active_users", "including_inactive_state"],
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_json}")


def main() -> None:
    args = parse_args()
    configure_style()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_state_share(input_dir, output_dir)
    plot_overlap(input_dir, output_dir)
    plot_mobility_index(input_dir, output_dir)
    plot_transition_heatmap(input_dir, output_dir)
    plot_overall_transition_matrix(input_dir, output_dir)


if __name__ == "__main__":
    main()
