#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_topic_by_media_type import TOPICS, load_json_data, normalize_text, topic_hits
from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "topic_stat_tests_by_media_type"
EXCLUDED_MONTHS = {"2025-11"}

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
TIER_ORDER = ["overall", "tier_2_5", "tier_6_20", "tier_21_50", "tier_51_plus"]
TIER_LABELS = {
    "overall": "Overall (all comments)",
    "tier_2_5": "2-5 comments",
    "tier_6_20": "6-20 comments",
    "tier_21_50": "21-50 comments",
    "tier_51_plus": "51+ comments",
}
TOPIC_ORDER = list(TOPICS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run media-type by topic chi-square tests overall and by commenter tier."
    )
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--exclude-month", action="append", default=sorted(EXCLUDED_MONTHS))
    parser.add_argument("--max-month", default="2025-10")
    return parser.parse_args()


def assign_tier(comment_count: int) -> str | None:
    if 2 <= comment_count <= 5:
        return "tier_2_5"
    if 6 <= comment_count <= 20:
        return "tier_6_20"
    if 21 <= comment_count <= 50:
        return "tier_21_50"
    if comment_count >= 51:
        return "tier_51_plus"
    return None


def load_user_tiers(path: Path) -> dict[str, str]:
    mapping = {}
    for row in load_json_data(path):
        user_id = row.get("authorChannelId") or row.get("user_id")
        if not user_id:
            continue
        tier = assign_tier(int(row.get("comment_count", 0) or 0))
        if tier:
            mapping[user_id] = tier
    return mapping


def load_video_media_types(path: Path) -> dict[str, str]:
    mapping = {}
    for row in load_json_data(path):
        video_id = row.get("id") or row.get("video_id")
        snippet = row.get("snippet") or {}
        channel = snippet.get("channelTitle") or row.get("channel_title") or row.get("channelTitle") or ""
        if video_id:
            mapping[video_id] = media_type_for_channel(channel)
    return mapping


def chi_square_stat(matrix: np.ndarray):
    row_totals = matrix.sum(axis=1, keepdims=True)
    col_totals = matrix.sum(axis=0, keepdims=True)
    total = matrix.sum()
    expected = row_totals @ col_totals / total
    residuals = np.divide(
        matrix - expected,
        np.sqrt(expected),
        out=np.zeros_like(expected, dtype=float),
        where=expected != 0,
    )
    chi2 = float(np.divide((matrix - expected) ** 2, expected, out=np.zeros_like(expected), where=expected != 0).sum())
    return chi2, expected, residuals


def cramers_v(chi2: float, matrix: np.ndarray) -> float:
    total = matrix.sum()
    min_dim = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    return float((chi2 / (total * min_dim)) ** 0.5) if total and min_dim > 0 else 0.0


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), used for chi-square survival."""
    if x < 0 or a <= 0:
        raise ValueError("Invalid arguments for incomplete gamma.")
    if x == 0:
        return 1.0

    itmax = 1000
    eps = 3.0e-14
    fpmin = 1.0e-300
    gln = math.lgamma(a)

    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        total = delta
        for _ in range(itmax):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                lower_p = total * math.exp(-x + a * math.log(x) - gln)
                return max(0.0, min(1.0, 1.0 - lower_p))
        lower_p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - lower_p))

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))


def chi_square_p_value(chi2: float, df: int) -> float:
    return gammaincc(df / 2.0, chi2 / 2.0)


def format_p_value(p_value: float) -> str:
    if p_value == 0.0:
        return "<1e-300"
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running_minimum = 1.0
    for reverse_rank, (index, p_value) in enumerate(reversed(ranked), start=1):
        rank = count - reverse_rank + 1
        running_minimum = min(running_minimum, p_value * count / rank)
        adjusted[index] = min(1.0, running_minimum)
    return adjusted


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_months = set(args.exclude_month or [])

    user_tiers = load_user_tiers(Path(args.commenters_json))
    video_media_types = load_video_media_types(Path(args.videos_json))

    topic_counts = {
        tier: {
            media_type: Counter({topic: 0 for topic in TOPIC_ORDER})
            for media_type in MEDIA_TYPE_ORDER
        }
        for tier in TIER_ORDER
    }
    comment_counts = {
        tier: Counter({media_type: 0 for media_type in MEDIA_TYPE_ORDER})
        for tier in TIER_ORDER
    }
    comments_with_any_topic = {
        tier: Counter({media_type: 0 for media_type in MEDIA_TYPE_ORDER})
        for tier in TIER_ORDER
    }
    users_by_group = defaultdict(set)
    videos_by_group = defaultdict(set)
    skipped = Counter()
    total_records = 0

    with Path(args.comments_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            row = json.loads(line)
            month = row.get("month")
            if month and month > args.max_month:
                skipped["after_max_month"] += 1
                continue
            if month in excluded_months:
                skipped["excluded_month"] += 1
                continue

            tier = user_tiers.get(row.get("user_id"))
            if not tier:
                skipped["one_comment_or_missing_user_tier_excluded_from_tiers"] += 1

            video_id = row.get("video_id")
            media_type = video_media_types.get(video_id, "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                skipped["unknown_media_type"] += 1
                continue

            text = normalize_text(row.get("text", ""))
            if not text:
                skipped["empty_text"] += 1
                continue

            hits, _ = topic_hits(text)
            tiers_for_record = ["overall"]
            if tier:
                tiers_for_record.append(tier)
            for tier_key in tiers_for_record:
                comment_counts[tier_key][media_type] += 1
                users_by_group[(tier_key, media_type)].add(row.get("user_id"))
                videos_by_group[(tier_key, media_type)].add(video_id)
                if hits:
                    comments_with_any_topic[tier_key][media_type] += 1
                for topic in hits:
                    topic_counts[tier_key][media_type][topic] += 1

    summary_rows = []
    residual_rows = []
    share_rows = []

    for tier in TIER_ORDER:
        for topic in TOPIC_ORDER:
            traditional_total = comment_counts[tier]["traditional"]
            emerging_total = comment_counts[tier]["emerging"]
            if not traditional_total or not emerging_total:
                skipped[f"{tier}_empty_comment_group"] += 1
                continue

            traditional_present = topic_counts[tier]["traditional"][topic]
            emerging_present = topic_counts[tier]["emerging"][topic]
            matrix = np.array(
                [
                    [traditional_present, traditional_total - traditional_present],
                    [emerging_present, emerging_total - emerging_present],
                ],
                dtype=float,
            )
            chi2, expected, residuals = chi_square_stat(matrix)
            p_value = chi_square_p_value(chi2, 1)
            traditional_share = traditional_present / traditional_total
            emerging_share = emerging_present / emerging_total
            denominator = math.sqrt(
                matrix[0].sum() * matrix[1].sum() * matrix[:, 0].sum() * matrix[:, 1].sum()
            )
            phi = ((matrix[0, 0] * matrix[1, 1]) - (matrix[0, 1] * matrix[1, 0])) / denominator if denominator else 0.0
            odds_ratio = (
                (traditional_present + 0.5) * (matrix[1, 1] + 0.5)
                / ((matrix[0, 1] + 0.5) * (emerging_present + 0.5))
            )
            summary_rows.append(
                {
                    "test": "media_type_by_binary_topic_presence",
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "topic": topic,
                    "unit_of_analysis": "comment",
                    "traditional_comment_count": int(traditional_total),
                    "emerging_comment_count": int(emerging_total),
                    "traditional_topic_present": int(traditional_present),
                    "emerging_topic_present": int(emerging_present),
                    "traditional_topic_share": f"{traditional_share:.6f}",
                    "emerging_topic_share": f"{emerging_share:.6f}",
                    "difference_percentage_points": f"{100 * (traditional_share - emerging_share):.6f}",
                    "odds_ratio_traditional_vs_emerging": f"{odds_ratio:.6f}",
                    "chi_square": f"{chi2:.6f}",
                    "p_value": p_value,
                    "degrees_of_freedom": 1,
                    "phi": f"{phi:.6f}",
                    "traditional_unique_users": len(users_by_group[(tier, "traditional")]),
                    "emerging_unique_users": len(users_by_group[(tier, "emerging")]),
                    "traditional_video_count": len(videos_by_group[(tier, "traditional")]),
                    "emerging_video_count": len(videos_by_group[(tier, "emerging")]),
                }
            )
            for index, media_type in enumerate(MEDIA_TYPE_ORDER):
                observed = matrix[index, 0]
                other_index = 1 - index
                residual_rows.append(
                    {
                        "tier": tier,
                        "tier_label": TIER_LABELS[tier],
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "topic": topic,
                        "observed_topic_present": int(observed),
                        "expected_topic_present": f"{expected[index, 0]:.4f}",
                        "standardized_residual": f"{residuals[index, 0]:.4f}",
                    }
                )
                present_share = matrix[index, 0] / matrix[index].sum()
                comparison_share = matrix[other_index, 0] / matrix[other_index].sum()
                share_rows.append(
                    {
                        "tier": tier,
                        "tier_label": TIER_LABELS[tier],
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "topic": topic,
                        "topic_present": int(matrix[index, 0]),
                        "comment_total": int(matrix[index].sum()),
                        "topic_presence_share": f"{present_share:.6f}",
                        "comparison_media_type": MEDIA_TYPE_ORDER[other_index],
                        "comparison_topic_presence_share": f"{comparison_share:.6f}",
                        "difference_vs_comparison": f"{present_share - comparison_share:.6f}",
                    }
                )

    adjusted = bh_adjust([row["p_value"] for row in summary_rows])
    adjusted_lookup = {}
    for row, adjusted_p in zip(summary_rows, adjusted):
        row["p_value"] = format_p_value(row["p_value"])
        row["p_value_bh"] = format_p_value(adjusted_p)
        adjusted_lookup[(row["tier"], row["topic"])] = row["p_value_bh"]
    for row in residual_rows:
        row["p_value_bh"] = adjusted_lookup[(row["tier"], row["topic"])]

    summary_fields = [
        "test", "tier", "tier_label", "topic", "unit_of_analysis",
        "traditional_comment_count", "emerging_comment_count",
        "traditional_topic_present", "emerging_topic_present",
        "traditional_topic_share", "emerging_topic_share", "difference_percentage_points",
        "odds_ratio_traditional_vs_emerging", "chi_square", "p_value", "p_value_bh",
        "degrees_of_freedom", "phi", "traditional_unique_users", "emerging_unique_users",
        "traditional_video_count", "emerging_video_count",
    ]
    residual_fields = [
        "tier", "tier_label", "media_type", "media_type_label", "topic",
        "observed_topic_present", "expected_topic_present", "standardized_residual", "p_value_bh",
    ]
    share_fields = [
        "tier", "tier_label", "media_type", "media_type_label", "topic",
        "topic_present", "comment_total", "topic_presence_share", "comparison_media_type",
        "comparison_topic_presence_share", "difference_vs_comparison",
    ]

    write_csv(output_dir / "test_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "standardized_residuals.csv", residual_rows, residual_fields)
    write_csv(output_dir / "topic_presence_shares_by_media_type.csv", share_rows, share_fields)

    summary = {
        "generated_by": "analyze_topic_stat_tests_by_media_type.py",
        "comments_jsonl": str(args.comments_jsonl),
        "commenters_json": str(args.commenters_json),
        "videos_json": str(args.videos_json),
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "total_records": total_records,
        "skipped": dict(skipped),
        "media_type_order": MEDIA_TYPE_ORDER,
        "tier_order": TIER_ORDER,
        "topic_order": TOPIC_ORDER,
        "method_note": (
            "Topic categories are keyword-based and non-mutually exclusive. Each domain is tested "
            "separately as a binary comment-level present/absent outcome. Benjamini-Hochberg correction "
            "is applied across all topic-by-tier tests."
        ),
        "outputs": {
            "test_summary_csv": str(output_dir / "test_summary.csv"),
            "residuals_csv": str(output_dir / "standardized_residuals.csv"),
            "shares_csv": str(output_dir / "topic_presence_shares_by_media_type.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'test_summary.csv'}")
    print(f"Wrote {output_dir / 'standardized_residuals.csv'}")
    print(f"Wrote {output_dir / 'topic_presence_shares_by_media_type.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
