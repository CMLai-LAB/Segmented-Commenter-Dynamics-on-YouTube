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

from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "emotion_stat_tests_media_type_by_tier"
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
EMOTION_MAP = {
    "平淡語氣": "Neutral",
    "關切語調": "Concerned",
    "開心語調": "Happy",
    "憤怒語調": "Angry",
    "悲傷語調": "Sad",
    "疑問語調": "Questioning",
    "驚奇語調": "Surprised",
    "厭惡語調": "Disgusted",
}
EMOTION_ORDER = ["Neutral", "Concerned", "Happy", "Angry", "Sad", "Questioning", "Surprised", "Disgusted"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run media-type by emotion chi-square tests overall and by commenter tier."
    )
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--exclude-month", action="append", default=sorted(EXCLUDED_MONTHS))
    parser.add_argument("--max-month", default="2025-10")
    return parser.parse_args()


def load_json_data(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


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
    pearson_residuals = np.divide(
        matrix - expected,
        np.sqrt(expected),
        out=np.zeros_like(expected, dtype=float),
        where=expected != 0,
    )
    row_proportions = np.divide(row_totals, total, out=np.zeros_like(row_totals), where=total != 0)
    col_proportions = np.divide(col_totals, total, out=np.zeros_like(col_totals), where=total != 0)
    adjusted_denominator = np.sqrt(expected * (1.0 - row_proportions) * (1.0 - col_proportions))
    adjusted_residuals = np.divide(
        matrix - expected,
        adjusted_denominator,
        out=np.zeros_like(expected, dtype=float),
        where=adjusted_denominator != 0,
    )
    chi2 = float(np.divide((matrix - expected) ** 2, expected, out=np.zeros_like(expected), where=expected != 0).sum())
    return chi2, expected, pearson_residuals, adjusted_residuals


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


def normal_two_sided_p_value(z_value: float) -> float:
    """Two-sided normal approximation p value for adjusted standardized residuals."""
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Return monotone BH-adjusted p values and FDR rejection indicators."""
    if not p_values:
        return [], []
    count = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank, (original_index, p_value) in reversed(list(enumerate(ranked, start=1))):
        running_minimum = min(running_minimum, p_value * count / rank)
        adjusted[original_index] = min(1.0, running_minimum)
    rejected = [p_value <= alpha for p_value in adjusted]
    return adjusted, rejected


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

    emotion_counts = {
        tier: {
            media_type: Counter({emotion: 0 for emotion in EMOTION_ORDER})
            for media_type in MEDIA_TYPE_ORDER
        }
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

            emotion = EMOTION_MAP.get(row.get("emotion_label"))
            if emotion not in EMOTION_ORDER:
                skipped["unknown_emotion"] += 1
                continue

            tiers_for_record = ["overall"]
            if tier:
                tiers_for_record.append(tier)
            for tier_key in tiers_for_record:
                emotion_counts[tier_key][media_type][emotion] += 1
                users_by_group[(tier_key, media_type)].add(row.get("user_id"))
                videos_by_group[(tier_key, media_type)].add(video_id)

    summary_rows = []
    residual_rows = []
    posthoc_rows = []
    share_rows = []
    omnibus_raw_p_values = []

    for tier in TIER_ORDER:
        matrix = np.array(
            [[emotion_counts[tier][media_type][emotion] for emotion in EMOTION_ORDER] for media_type in MEDIA_TYPE_ORDER],
            dtype=float,
        )
        if matrix.sum() == 0:
            skipped[f"{tier}_empty_emotion_matrix"] += 1
            continue

        chi2, expected, pearson_residuals, adjusted_residuals = chi_square_stat(matrix)
        df = (len(MEDIA_TYPE_ORDER) - 1) * (len(EMOTION_ORDER) - 1)
        p_value = chi_square_p_value(chi2, df)
        omnibus_raw_p_values.append(p_value)

        summary_rows.append(
            {
                "test": "media_type_by_emotion_distribution",
                "tier": tier,
                "tier_label": TIER_LABELS[tier],
                "unit_of_analysis": "comment",
                "traditional_comment_count": int(matrix[0].sum()),
                "emerging_comment_count": int(matrix[1].sum()),
                "traditional_unique_users": len(users_by_group[(tier, "traditional")]),
                "emerging_unique_users": len(users_by_group[(tier, "emerging")]),
                "traditional_video_count": len(videos_by_group[(tier, "traditional")]),
                "emerging_video_count": len(videos_by_group[(tier, "emerging")]),
                "chi_square": f"{chi2:.6f}",
                "p_value": format_p_value(p_value),
                "p_value_raw": format_p_value(p_value),
                "p_value_bh": "",
                "reject_bh_fdr_0_05": "",
                "degrees_of_freedom": df,
                "cramers_v": f"{cramers_v(chi2, matrix):.6f}",
            }
        )

        for i, media_type in enumerate(MEDIA_TYPE_ORDER):
            row_total = matrix[i].sum()
            for j, emotion in enumerate(EMOTION_ORDER):
                observed = matrix[i, j]
                other_media_type = MEDIA_TYPE_ORDER[1 - i]
                other_total = matrix[1 - i].sum()
                other_observed = matrix[1 - i, j]
                share = observed / row_total if row_total else 0.0
                other_share = other_observed / other_total if other_total else 0.0
                residual_rows.append(
                    {
                        "tier": tier,
                        "tier_label": TIER_LABELS[tier],
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "emotion": emotion,
                        "observed_comments": int(observed),
                        "expected_comments": f"{expected[i, j]:.4f}",
                        "pearson_residual": f"{pearson_residuals[i, j]:.4f}",
                        "adjusted_standardized_residual": f"{adjusted_residuals[i, j]:.4f}",
                        "standardized_residual": f"{adjusted_residuals[i, j]:.4f}",
                        "residual_p_value_raw": "",
                        "residual_p_value_bh": "",
                        "residual_reject_bh_fdr_0_05": "",
                    }
                )
                share_rows.append(
                    {
                        "tier": tier,
                        "tier_label": TIER_LABELS[tier],
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "emotion": emotion,
                        "comment_count": int(observed),
                        "media_type_comment_total": int(row_total),
                        "emotion_share": f"{share:.6f}",
                        "comparison_media_type": other_media_type,
                        "comparison_emotion_share": f"{other_share:.6f}",
                        "difference_vs_comparison": f"{share - other_share:.6f}",
                    }
                )

                # A two-row media-type table produces mirror residuals. Test each tier-emotion
                # contrast once, in the traditional-media direction, then attach the result to both rows.
                if media_type == "traditional":
                    posthoc_rows.append(
                        {
                            "tier": tier,
                            "tier_label": TIER_LABELS[tier],
                            "emotion": emotion,
                            "traditional_comment_count": int(observed),
                            "emerging_comment_count": int(other_observed),
                            "traditional_emotion_share": f"{share:.6f}",
                            "emerging_emotion_share": f"{other_share:.6f}",
                            "share_difference_traditional_minus_emerging": f"{share - other_share:.6f}",
                            "adjusted_standardized_residual": f"{adjusted_residuals[i, j]:.4f}",
                            "p_value_raw": format_p_value(normal_two_sided_p_value(adjusted_residuals[i, j])),
                            "p_value_bh": "",
                            "reject_bh_fdr_0_05": "",
                        }
                    )

    omnibus_bh_values, omnibus_rejections = benjamini_hochberg(omnibus_raw_p_values)
    for row, adjusted_p_value, rejected in zip(summary_rows, omnibus_bh_values, omnibus_rejections):
        row["p_value_bh"] = format_p_value(adjusted_p_value)
        row["reject_bh_fdr_0_05"] = str(rejected)

    posthoc_raw_p_values = [
        normal_two_sided_p_value(float(row["adjusted_standardized_residual"]))
        for row in posthoc_rows
    ]
    posthoc_bh_values, posthoc_rejections = benjamini_hochberg(posthoc_raw_p_values)
    posthoc_lookup = {}
    for row, raw_p_value, adjusted_p_value, rejected in zip(
        posthoc_rows,
        posthoc_raw_p_values,
        posthoc_bh_values,
        posthoc_rejections,
    ):
        row["p_value_raw"] = format_p_value(raw_p_value)
        row["p_value_bh"] = format_p_value(adjusted_p_value)
        row["reject_bh_fdr_0_05"] = str(rejected)
        posthoc_lookup[(row["tier"], row["emotion"])] = (raw_p_value, adjusted_p_value, rejected)

    for row in residual_rows:
        raw_p_value, adjusted_p_value, rejected = posthoc_lookup[(row["tier"], row["emotion"])]
        row["residual_p_value_raw"] = format_p_value(raw_p_value)
        row["residual_p_value_bh"] = format_p_value(adjusted_p_value)
        row["residual_reject_bh_fdr_0_05"] = str(rejected)

    summary_fields = [
        "test",
        "tier",
        "tier_label",
        "unit_of_analysis",
        "traditional_comment_count",
        "emerging_comment_count",
        "traditional_unique_users",
        "emerging_unique_users",
        "traditional_video_count",
        "emerging_video_count",
        "chi_square",
        "p_value",
        "p_value_raw",
        "p_value_bh",
        "reject_bh_fdr_0_05",
        "degrees_of_freedom",
        "cramers_v",
    ]
    residual_fields = [
        "tier",
        "tier_label",
        "media_type",
        "media_type_label",
        "emotion",
        "observed_comments",
        "expected_comments",
        "pearson_residual",
        "adjusted_standardized_residual",
        "standardized_residual",
        "residual_p_value_raw",
        "residual_p_value_bh",
        "residual_reject_bh_fdr_0_05",
    ]
    share_fields = [
        "tier",
        "tier_label",
        "media_type",
        "media_type_label",
        "emotion",
        "comment_count",
        "media_type_comment_total",
        "emotion_share",
        "comparison_media_type",
        "comparison_emotion_share",
        "difference_vs_comparison",
    ]

    write_csv(output_dir / "test_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "standardized_residuals.csv", residual_rows, residual_fields)
    write_csv(output_dir / "posthoc_adjusted_residual_tests.csv", posthoc_rows)
    write_csv(output_dir / "emotion_shares_by_media_type.csv", share_rows, share_fields)

    summary = {
        "generated_by": "analyze_emotion_media_type_by_tier_chi_square.py",
        "comments_jsonl": str(args.comments_jsonl),
        "commenters_json": str(args.commenters_json),
        "videos_json": str(args.videos_json),
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "total_records": total_records,
        "skipped": dict(skipped),
        "media_type_order": MEDIA_TYPE_ORDER,
        "tier_order": TIER_ORDER,
        "emotion_order": EMOTION_ORDER,
        "method_note": (
            "Emotion labels are mutually exclusive per comment. "
            "The chi-square tests compare the distribution of emotion labels between media types "
            "overall and within each commenter tier. Omnibus p values are Benjamini-Hochberg adjusted over "
            "the five overall-or-tier tests. Post-hoc contrasts use adjusted standardized residuals; "
            "two-sided normal-approximation p values are Benjamini-Hochberg adjusted over the "
            "40 non-duplicated tier-by-emotion contrasts."
        ),
        "multiple_testing": {
            "method": "Benjamini-Hochberg false discovery rate",
            "alpha": 0.05,
            "omnibus_test_count": len(summary_rows),
            "posthoc_test_count": len(posthoc_rows),
            "posthoc_unit": "one traditional-minus-emerging contrast per tier and emotion",
        },
        "outputs": {
            "test_summary_csv": str(output_dir / "test_summary.csv"),
            "residuals_csv": str(output_dir / "standardized_residuals.csv"),
            "posthoc_tests_csv": str(output_dir / "posthoc_adjusted_residual_tests.csv"),
            "shares_csv": str(output_dir / "emotion_shares_by_media_type.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'test_summary.csv'}")
    print(f"Wrote {output_dir / 'standardized_residuals.csv'}")
    print(f"Wrote {output_dir / 'posthoc_adjusted_residual_tests.csv'}")
    print(f"Wrote {output_dir / 'emotion_shares_by_media_type.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
