#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "emotion_stat_tests_by_media_type"
EXCLUDED_MONTHS = {"2025-11"}

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
TIER_ORDER = ["tier_2_5", "tier_6_20", "tier_21_50", "tier_51_plus"]
TIER_LABELS = {
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
    parser = argparse.ArgumentParser(description="Run tier-emotion residual tests separately by media type.")
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


def load_video_media_types(path: Path):
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
    residuals = np.divide(matrix - expected, np.sqrt(expected), out=np.zeros_like(expected, dtype=float), where=expected != 0)
    chi2 = float(((matrix - expected) ** 2 / expected).sum())
    return chi2, expected, residuals


def cramers_v(chi2: float, matrix: np.ndarray) -> float:
    total = matrix.sum()
    min_dim = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    return float((chi2 / (total * min_dim)) ** 0.5) if total and min_dim > 0 else 0.0


def p_label(chi2: float) -> str:
    if chi2 > 1500:
        return "<1e-300"
    if chi2 > 1000:
        return "<1e-200"
    if chi2 > 500:
        return "<1e-100"
    if chi2 > 300:
        return "<1e-50"
    return "not_computed"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
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
    counts = {
        media_type: {
            tier: Counter({emotion: 0 for emotion in EMOTION_ORDER})
            for tier in TIER_ORDER
        }
        for media_type in MEDIA_TYPE_ORDER
    }
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
                skipped["one_comment_or_missing_user_tier"] += 1
                continue
            media_type = video_media_types.get(row.get("video_id"), "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                skipped["unknown_media_type"] += 1
                continue
            emotion = EMOTION_MAP.get(row.get("emotion_label"))
            if emotion not in EMOTION_ORDER:
                skipped["unknown_emotion"] += 1
                continue
            counts[media_type][tier][emotion] += 1

    summary_rows = []
    residual_rows = []
    share_rows = []

    for media_type in MEDIA_TYPE_ORDER:
        matrix = np.array(
            [[counts[media_type][tier][emotion] for emotion in EMOTION_ORDER] for tier in TIER_ORDER],
            dtype=float,
        )
        chi2, expected, residuals = chi_square_stat(matrix)
        df = (len(TIER_ORDER) - 1) * (len(EMOTION_ORDER) - 1)
        summary_rows.append(
            {
                "media_type": media_type,
                "media_type_label": MEDIA_TYPE_LABELS[media_type],
                "comment_count": int(matrix.sum()),
                "chi_square": f"{chi2:.6f}",
                "p_value": p_label(chi2),
                "degrees_of_freedom": df,
                "cramers_v": f"{cramers_v(chi2, matrix):.6f}",
            }
        )
        for i, tier in enumerate(TIER_ORDER):
            row_total = matrix[i].sum()
            for j, emotion in enumerate(EMOTION_ORDER):
                observed = matrix[i, j]
                residual_rows.append(
                    {
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "row": tier,
                        "tier_label": TIER_LABELS[tier],
                        "emotion": emotion,
                        "observed": int(observed),
                        "expected": f"{expected[i, j]:.4f}",
                        "standardized_residual": f"{residuals[i, j]:.4f}",
                    }
                )
                share_rows.append(
                    {
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "row": tier,
                        "tier_label": TIER_LABELS[tier],
                        "emotion": emotion,
                        "count": int(observed),
                        "row_total": int(row_total),
                        "row_share": f"{(observed / row_total if row_total else 0):.6f}",
                    }
                )

        media_residual_rows = [row for row in residual_rows if row["media_type"] == media_type]
        media_share_rows = [row for row in share_rows if row["media_type"] == media_type]
        write_csv(output_dir / f"{media_type}_tier_standardized_residuals.csv", media_residual_rows)
        write_csv(output_dir / f"{media_type}_tier_emotion_shares.csv", media_share_rows)

    write_csv(output_dir / "test_summary.csv", summary_rows)
    write_csv(output_dir / "tier_standardized_residuals_by_media_type.csv", residual_rows)
    write_csv(output_dir / "tier_emotion_shares_by_media_type.csv", share_rows)

    summary = {
        "generated_by": "analyze_emotion_stat_tests_by_media_type.py",
        "total_records": total_records,
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "skipped": dict(skipped),
        "emotion_order": EMOTION_ORDER,
        "tier_order": TIER_ORDER,
        "outputs": {
            "test_summary_csv": str(output_dir / "test_summary.csv"),
            "residuals_csv": str(output_dir / "tier_standardized_residuals_by_media_type.csv"),
            "shares_csv": str(output_dir / "tier_emotion_shares_by_media_type.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'test_summary.csv'}")
    print(f"Wrote {output_dir / 'tier_standardized_residuals_by_media_type.csv'}")
    print(f"Wrote {output_dir / 'tier_emotion_shares_by_media_type.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
