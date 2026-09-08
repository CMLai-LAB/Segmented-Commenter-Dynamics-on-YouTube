#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "user_mobility_by_media_type" / "quarterly_user_transition_churn_summary.csv"
OUTPUT_DIR = BASE_DIR / "user_mobility_by_media_type"
PAPER_OUTPUT_DIR = BASE_DIR / "paper_statistical_figures"

STATE_ORDER = ["traditional_only", "emerging_only", "both", "inactive"]
STATE_LABELS = {
    "traditional_only": "Traditional",
    "emerging_only": "Emerging",
    "both": "Both",
    "inactive": "Inactive",
}
STATE_SHORT_LABELS = {
    "traditional_only": "T",
    "emerging_only": "E",
    "both": "Both",
    "inactive": "Inactive",
}
MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "mobility_matrix",
    ["#F7F9FB", "#DDE8F1", "#AFC8DC", "#6F9BC1", "#2F638F"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quarterly media-type user transition matrix panels.")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--paper-output-dir", default=PAPER_OUTPUT_DIR)
    parser.add_argument("--annotate-threshold", type=float, default=0.05)
    parser.add_argument("--clean", action="store_true", help="Do not annotate percentages inside cells.")
    parser.add_argument(
        "--hide-metrics",
        action="store_true",
        help="Do not show retention, churn, and bridge summaries below each matrix.",
    )
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


def read_pairs(path: Path) -> dict[str, dict]:
    pairs = defaultdict(
        lambda: {
            "from_quarter": "",
            "to_quarter": "",
            "counts": Counter(),
            "row_shares": {},
            "pair_shares": {},
            "union_user_count": 0,
        }
    )
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair = row["quarter_pair"]
            from_state = row["from_state"]
            to_state = row["to_state"]
            pairs[pair]["from_quarter"] = row["from_quarter"]
            pairs[pair]["to_quarter"] = row["to_quarter"]
            pairs[pair]["counts"][(from_state, to_state)] += int(row["user_count"])
            pairs[pair]["row_shares"][(from_state, to_state)] = float(row["row_share"])
            pairs[pair]["pair_shares"][(from_state, to_state)] = float(row["pair_share"])
            pairs[pair]["union_user_count"] = int(row["union_user_count"])
    return dict(sorted(pairs.items()))


def active_from_total(counts: Counter) -> int:
    return sum(
        counts[(source, target)]
        for source in STATE_ORDER
        if source != "inactive"
        for target in STATE_ORDER
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mobility_metrics(counts: Counter) -> dict:
    active_total = active_from_total(counts)
    retention_count = sum(counts[(state, state)] for state in ["traditional_only", "emerging_only", "both"])
    churn_count = sum(counts[(state, "inactive")] for state in ["traditional_only", "emerging_only", "both"])
    direct_switch_count = (
        counts[("traditional_only", "emerging_only")]
        + counts[("emerging_only", "traditional_only")]
    )
    bridge_count = (
        counts[("traditional_only", "both")]
        + counts[("emerging_only", "both")]
    )
    return {
        "active_from_user_count": active_total,
        "retention_count": retention_count,
        "retention_rate": safe_ratio(retention_count, active_total),
        "churn_count": churn_count,
        "churn_rate": safe_ratio(churn_count, active_total),
        "direct_switch_count": direct_switch_count,
        "direct_switch_rate": safe_ratio(direct_switch_count, active_total),
        "bridge_count": bridge_count,
        "bridge_rate": safe_ratio(bridge_count, active_total),
    }


def matrix_from_pair(pair_data: dict) -> list[list[float]]:
    return [
        [pair_data["row_shares"].get((source, target), 0.0) for target in STATE_ORDER]
        for source in STATE_ORDER
    ]


def draw_panel(
    ax,
    pair: str,
    pair_data: dict,
    annotate: bool,
    annotate_threshold: float,
    show_metrics: bool,
) -> dict:
    matrix = matrix_from_pair(pair_data)
    # The adjacent-period union cannot contain an inactive-to-inactive user.
    inactive_index = STATE_ORDER.index("inactive")
    mask = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=bool)
    mask[inactive_index, inactive_index] = True
    plot_matrix = np.ma.array(matrix, mask=mask)
    cmap = MATRIX_CMAP.copy()
    cmap.set_bad("#E5E7EB")
    image = ax.imshow(plot_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.text(inactive_index, inactive_index, "N/A", ha="center", va="center",
            fontsize=8.1, color="#6B7280")

    ax.set_title(
        f"{pair_data['from_quarter']} -> {pair_data['to_quarter']}",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    ax.set_xticks(range(len(STATE_ORDER)))
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_xticklabels([STATE_SHORT_LABELS[state] for state in STATE_ORDER], fontsize=8.6)
    ax.set_yticklabels([STATE_SHORT_LABELS[state] for state in STATE_ORDER], fontsize=8.6)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks([x - 0.5 for x in range(1, len(STATE_ORDER))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(STATE_ORDER))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for row_idx, source in enumerate(STATE_ORDER):
            for col_idx, target in enumerate(STATE_ORDER):
                if source == target == "inactive":
                    continue
                value = matrix[row_idx][col_idx]
                count = pair_data["counts"].get((source, target), 0)
                if value < annotate_threshold and count > 0 and source != target:
                    continue
                if count == 0:
                    continue
                color = "white" if value >= 0.55 else "#1F2933"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.1,
                    fontweight="bold" if value >= 0.25 else "normal",
                    color=color,
                )

    metrics = mobility_metrics(pair_data["counts"])
    if show_metrics:
        ax.text(
            0.5,
            -0.17,
            f"Retention {metrics['retention_rate']:.1%} | Churn {metrics['churn_rate']:.1%} | Bridge {metrics['bridge_rate']:.1%}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#4A4A4A",
        )
    return {
        "quarter_pair": pair,
        "from_quarter": pair_data["from_quarter"],
        "to_quarter": pair_data["to_quarter"],
        "union_user_count": pair_data["union_user_count"],
        "matrix_row_shares": {
            source: {
                target: pair_data["row_shares"].get((source, target), 0.0)
                for target in STATE_ORDER
            }
            for source in STATE_ORDER
        },
        "metrics": metrics,
    }, image


def draw_matrix_panels(
    pairs: dict[str, dict],
    input_csv: Path,
    output_png: Path,
    output_json: Path,
    *,
    annotate: bool,
    annotate_threshold: float,
    show_metrics: bool,
) -> None:
    pair_items = list(pairs.items())
    cols = 3
    rows = (len(pair_items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15.2, max(8.4, rows * 3.8)), facecolor="white")
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    summaries = []
    image = None
    for ax, (pair, pair_data) in zip(flat_axes, pair_items):
        summary, image = draw_panel(
            ax,
            pair,
            pair_data,
            annotate,
            annotate_threshold,
            show_metrics,
        )
        summaries.append(summary)
    for ax in flat_axes[len(pair_items):]:
        ax.axis("off")

    fig.supylabel("From previous quarter", fontsize=10.8, x=0.012)
    if image is not None:
        cax = fig.add_axes([0.910, 0.185, 0.014, 0.640])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label("Row-normalized transition share", fontsize=9.3)
        colorbar.ax.tick_params(labelsize=8.5)

    bottom = 0.100 if show_metrics else 0.075
    hspace = 0.58 if show_metrics else 0.40
    fig.subplots_adjust(left=0.070, right=0.885, top=0.935, bottom=bottom, hspace=hspace, wspace=0.34)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output_png}")

    summary = {
        "generated_by": Path(__file__).name,
        "input_csv": str(input_csv),
        "output_png": str(output_png),
        "annotate": annotate,
        "annotate_threshold": annotate_threshold,
        "show_metrics": show_metrics,
        "states": STATE_LABELS,
        "note": "Each heatmap is row-normalized within the cohort with at least two in-corpus comments over the full observation window. Rows indicate the previous-quarter state; columns indicate the next-quarter state. The structurally impossible inactive-to-inactive cell is masked and marked N/A.",
        "panels": summaries,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")


def main() -> None:
    args = parse_args()
    configure_style()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    paper_output_dir = Path(args.paper_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(input_csv)

    annotate = not args.clean
    suffix = "" if annotate else "_clean"
    output_png = output_dir / f"quarterly_media_type_transition_matrix_panels{suffix}.png"
    output_json = output_dir / f"quarterly_media_type_transition_matrix_panels{suffix}_summary.json"
    draw_matrix_panels(
        pairs,
        input_csv,
        output_png,
        output_json,
        annotate=annotate,
        annotate_threshold=args.annotate_threshold,
        show_metrics=not args.hide_metrics,
    )

    paper_png = paper_output_dir / f"quarterly_media_type_transition_matrix_panels{suffix}.png"
    paper_json = paper_output_dir / f"quarterly_media_type_transition_matrix_panels{suffix}_summary.json"
    draw_matrix_panels(
        pairs,
        input_csv,
        paper_png,
        paper_json,
        annotate=annotate,
        annotate_threshold=args.annotate_threshold,
        show_metrics=not args.hide_metrics,
    )


if __name__ == "__main__":
    main()
