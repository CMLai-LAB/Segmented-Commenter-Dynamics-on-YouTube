#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MEDIA_TYPE_DIR = BASE_DIR / "user_mobility_by_media_type"
CHANNEL_DIR = BASE_DIR / "user_mobility_by_channel"
OUTPUT_DIR = BASE_DIR / "user_mobility_bootstrap_ci"


METRIC_SPECS = [
    {
        "analysis_scope": "media_type_monthly_state",
        "source_file": MEDIA_TYPE_DIR / "monthly_user_media_state_summary.csv",
        "group_columns": ["state"],
        "label_columns": ["state_label"],
        "bootstrap_unit": "month",
        "metrics": [
            ("user_share", "user_count", "active_user_count"),
            ("comment_share", "comment_count", "active_comment_count"),
        ],
    },
    {
        "analysis_scope": "media_type_monthly_overlap",
        "source_file": MEDIA_TYPE_DIR / "monthly_user_overlap_summary.csv",
        "group_columns": [],
        "label_columns": [],
        "bootstrap_unit": "month",
        "metrics": [
            ("jaccard_overlap", "overlap_user_count", "union_user_count"),
            ("overlap_share_of_traditional", "overlap_user_count", "traditional_user_count"),
            ("overlap_share_of_emerging", "overlap_user_count", "emerging_user_count"),
        ],
    },
    {
        "analysis_scope": "media_type_monthly_transition",
        "source_file": MEDIA_TYPE_DIR / "monthly_user_mobility_index.csv",
        "group_columns": [],
        "label_columns": [],
        "bootstrap_unit": "month_pair",
        "metrics": [
            ("same_state_share", "same_state_count", "continuing_user_count"),
            ("state_change_share", "state_change_count", "continuing_user_count"),
            ("side_switch_share", "side_switch_count", "continuing_user_count"),
            ("single_to_both_share", "single_to_both_count", "continuing_user_count"),
            ("both_to_single_share", "both_to_single_count", "continuing_user_count"),
            ("both_to_both_share", "both_to_both_count", "continuing_user_count"),
            ("active_to_inactive_share_of_union", "active_to_inactive_count", "union_user_count"),
            ("inactive_to_active_share_of_union", "inactive_to_active_count", "union_user_count"),
            ("retained_share_of_union", "retained_user_count", "union_user_count"),
        ],
    },
    {
        "analysis_scope": "channel_monthly_breadth",
        "source_file": CHANNEL_DIR / "monthly_user_channel_breadth_summary.csv",
        "group_columns": ["channel_count_bucket"],
        "label_columns": ["channel_count_bucket_label"],
        "bootstrap_unit": "month",
        "metrics": [
            ("user_share", "user_count", "active_user_count"),
            ("comment_share", "comment_count", "active_comment_count"),
        ],
    },
    {
        "analysis_scope": "channel_monthly_transition_type",
        "source_file": CHANNEL_DIR / "monthly_channel_transition_type_summary.csv",
        "group_columns": ["transition_type"],
        "label_columns": ["transition_type_label"],
        "bootstrap_unit": "month_pair",
        "metrics": [
            ("continuing_user_share", "user_count", "continuing_user_count"),
        ],
    },
    {
        "analysis_scope": "channel_monthly_transition",
        "source_file": CHANNEL_DIR / "monthly_channel_mobility_index.csv",
        "group_columns": [],
        "label_columns": [],
        "bootstrap_unit": "month_pair",
        "metrics": [
            ("same_channel_share", "same_channel_count", "continuing_user_count"),
            ("same_media_different_channel_share", "same_media_different_channel_count", "continuing_user_count"),
            ("cross_media_share", "cross_media_count", "continuing_user_count"),
            ("changed_channel_share", "changed_channel_count", "continuing_user_count"),
            ("inactive_next_month_share_of_from_active", "inactive_next_month_count", "from_active_user_count"),
            ("retained_next_month_share_of_from_active", "retained_next_month_count", "from_active_user_count"),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for monthly user mobility metrics."
    )
    parser.add_argument("--media-type-dir", default=MEDIA_TYPE_DIR)
    parser.add_argument("--channel-dir", default=CHANNEL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def group_key(row: dict, columns: list[str]) -> tuple:
    return tuple(row[column] for column in columns)


def label_for_group(rows: list[dict], columns: list[str]) -> dict[str, str]:
    if not rows:
        return {}
    return {column: rows[0].get(column, "") for column in columns}


def ratio_from_rows(rows: list[dict], numerator_column: str, denominator_column: str) -> float:
    numerator = sum(as_float(row[numerator_column]) for row in rows)
    denominator = sum(as_float(row[denominator_column]) for row in rows)
    return numerator / denominator if denominator else 0.0


def bootstrap_ratio_ci(
    rows: list[dict],
    numerator_column: str,
    denominator_column: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0

    numerator = np.array([as_float(row[numerator_column]) for row in rows], dtype=float)
    denominator = np.array([as_float(row[denominator_column]) for row in rows], dtype=float)
    n_units = len(rows)
    estimates = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_units, size=n_units)
        sample_denominator = denominator[sample_idx].sum()
        estimates[i] = numerator[sample_idx].sum() / sample_denominator if sample_denominator else 0.0

    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return float(estimates.mean()), lower, upper


def build_rows_for_spec(spec: dict, n_bootstrap: int, alpha: float, rng: np.random.Generator) -> list[dict]:
    rows = read_csv(Path(spec["source_file"]))
    grouped = defaultdict(list)
    if spec["group_columns"]:
        for row in rows:
            grouped[group_key(row, spec["group_columns"])].append(row)
    else:
        grouped[()].extend(rows)

    output_rows = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        group_values = dict(zip(spec["group_columns"], key))
        label_values = label_for_group(group_rows, spec["label_columns"])

        for metric, numerator_column, denominator_column in spec["metrics"]:
            usable_rows = [row for row in group_rows if as_float(row.get(denominator_column, 0)) > 0]
            observed = ratio_from_rows(usable_rows, numerator_column, denominator_column)
            bootstrap_mean, ci_lower, ci_upper = bootstrap_ratio_ci(
                usable_rows,
                numerator_column,
                denominator_column,
                n_bootstrap,
                alpha,
                rng,
            )
            output_rows.append(
                {
                    "analysis_scope": spec["analysis_scope"],
                    "source_csv": str(spec["source_file"]),
                    "bootstrap_unit": spec["bootstrap_unit"],
                    "metric": metric,
                    "group": "overall" if not group_values else "; ".join(
                        f"{column}={value}" for column, value in group_values.items()
                    ),
                    **group_values,
                    **label_values,
                    "numerator_column": numerator_column,
                    "denominator_column": denominator_column,
                    "observed_value": f"{observed:.6f}",
                    "bootstrap_mean": f"{bootstrap_mean:.6f}",
                    "ci_lower": f"{ci_lower:.6f}",
                    "ci_upper": f"{ci_upper:.6f}",
                    "n_units": len(usable_rows),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return output_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    media_type_dir = Path(args.media_type_dir)
    channel_dir = Path(args.channel_dir)
    specs = []
    for spec in METRIC_SPECS:
        spec = dict(spec)
        source_file = Path(spec["source_file"])
        if source_file.parts[-2] == "user_mobility_by_media_type":
            spec["source_file"] = media_type_dir / source_file.name
        elif source_file.parts[-2] == "user_mobility_by_channel":
            spec["source_file"] = channel_dir / source_file.name
        specs.append(spec)

    rng = np.random.default_rng(args.seed)
    alpha = 1.0 - args.confidence
    all_rows = []
    skipped_sources = []

    for spec in specs:
        if not Path(spec["source_file"]).exists():
            skipped_sources.append(str(spec["source_file"]))
            continue
        all_rows.extend(build_rows_for_spec(spec, args.n_bootstrap, alpha, rng))

    fieldnames = [
        "analysis_scope",
        "source_csv",
        "bootstrap_unit",
        "metric",
        "group",
        "state",
        "state_label",
        "channel_count_bucket",
        "channel_count_bucket_label",
        "transition_type",
        "transition_type_label",
        "numerator_column",
        "denominator_column",
        "observed_value",
        "bootstrap_mean",
        "ci_lower",
        "ci_upper",
        "n_units",
        "n_bootstrap",
    ]
    for row in all_rows:
        for fieldname in fieldnames:
            row.setdefault(fieldname, "")

    all_output = output_dir / "user_mobility_bootstrap_ci.csv"
    media_output = output_dir / "media_type_mobility_bootstrap_ci.csv"
    channel_output = output_dir / "channel_mobility_bootstrap_ci.csv"

    write_csv(all_output, all_rows, fieldnames)
    write_csv(
        media_output,
        [row for row in all_rows if row["analysis_scope"].startswith("media_type")],
        fieldnames,
    )
    write_csv(
        channel_output,
        [row for row in all_rows if row["analysis_scope"].startswith("channel")],
        fieldnames,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": Path(__file__).name,
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "method_note": (
            "Bootstrap resamples months for monthly state/overlap/breadth metrics and month-pairs "
            "for transition metrics. Each bootstrap estimate is a weighted ratio, computed as the "
            "sum of numerator counts divided by the sum of denominator counts in the resampled units."
        ),
        "skipped_sources": skipped_sources,
        "outputs": {
            "all_ci_csv": str(all_output),
            "media_type_ci_csv": str(media_output),
            "channel_ci_csv": str(channel_output),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {all_output}")
    print(f"Wrote {media_output}")
    print(f"Wrote {channel_output}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
