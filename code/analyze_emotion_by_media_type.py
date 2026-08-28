#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "emotion_by_media_type"
EXCLUDED_MONTHS = {"2025-11"}

EMOTION_LABELS_ZH = [
    "平淡語氣",
    "關切語調",
    "開心語調",
    "憤怒語調",
    "悲傷語調",
    "疑問語調",
    "驚奇語調",
    "厭惡語調",
]

EMOTION_LABELS_EN = {
    "平淡語氣": "Neutral",
    "關切語調": "Concerned",
    "開心語調": "Happy",
    "憤怒語調": "Angry",
    "悲傷語調": "Sad",
    "疑問語調": "Questioning",
    "驚奇語調": "Surprised",
    "厭惡語調": "Disgusted",
}

EMOTION_ORDER = [EMOTION_LABELS_EN[label] for label in EMOTION_LABELS_ZH]
MEDIA_TYPE_ORDER = ["traditional", "emerging"]
TIER_ORDER = ["overall", "low_2_5", "mid_6_20", "core_21_50", "high_51_plus"]
TIER_LABELS = {
    "overall": "Overall (all comments)",
    "low_2_5": "2-5 comments",
    "mid_6_20": "6-20 comments",
    "core_21_50": "21-50 comments",
    "high_51_plus": "51+ comments",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate comment emotions by media type and commenter tier.")
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
    if comment_count <= 1:
        return None
    if comment_count <= 5:
        return "low_2_5"
    if comment_count <= 20:
        return "mid_6_20"
    if comment_count <= 50:
        return "core_21_50"
    return "high_51_plus"


def load_user_tiers(path: Path) -> dict[str, str]:
    mapping = {}
    for row in load_json_data(path):
        user_id = row.get("authorChannelId")
        if not user_id:
            continue
        tier = assign_tier(int(row.get("comment_count", 0) or 0))
        if tier:
            mapping[user_id] = tier
    return mapping


def parse_count(value) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_video_metadata(path: Path):
    video_media_type = {}
    video_stats = {}
    channel_counts = Counter()
    media_counts = Counter()
    media_view_counts = Counter()
    media_video_comment_counts = Counter()
    for video in load_json_data(path):
        video_id = video.get("id")
        snippet = video.get("snippet") or {}
        statistics = video.get("statistics") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        media_type = media_type_for_channel(channel)
        if not video_id:
            continue
        view_count = parse_count(statistics.get("viewCount"))
        video_comment_count = parse_count(statistics.get("commentCount"))
        video_media_type[video_id] = media_type
        video_stats[video_id] = {
            "view_count": view_count,
            "video_comment_count": video_comment_count,
        }
        channel_counts[channel] += 1
        media_counts[media_type] += 1
        media_view_counts[media_type] += view_count
        media_video_comment_counts[media_type] += video_comment_count
    return (
        video_media_type,
        video_stats,
        channel_counts,
        media_counts,
        media_view_counts,
        media_video_comment_counts,
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def per_1000(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator * 1000 if denominator else 0.0


def log2_ratio(a_value: float, b_value: float, smoothing: float = 1e-9) -> float:
    import math

    return math.log2((a_value + smoothing) / (b_value + smoothing))


def sum_video_metric(video_ids: set[str], video_stats: dict[str, dict], metric: str) -> int:
    return sum(video_stats.get(video_id, {}).get(metric, 0) for video_id in video_ids)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_months = set(args.exclude_month or [])

    user_to_tier = load_user_tiers(Path(args.commenters_json))
    (
        video_media_type,
        video_stats,
        channel_video_counts,
        media_video_counts,
        media_view_counts,
        media_video_comment_counts,
    ) = load_video_metadata(Path(args.videos_json))

    counts = {
        media_type: {tier: Counter() for tier in TIER_ORDER}
        for media_type in MEDIA_TYPE_ORDER
    }
    users = {
        media_type: {tier: set() for tier in TIER_ORDER}
        for media_type in MEDIA_TYPE_ORDER
    }
    videos = {
        media_type: {tier: set() for tier in TIER_ORDER}
        for media_type in MEDIA_TYPE_ORDER
    }
    monthly_counts = defaultdict(Counter)
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
            tier = user_to_tier.get(row.get("user_id"))
            if not tier:
                skipped["one_comment_or_missing_user_tier_excluded_from_tiers"] += 1
            media_type = video_media_type.get(row.get("video_id"), "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                skipped["unknown_media_type"] += 1
                continue
            emotion_zh = row.get("emotion_label")
            if emotion_zh not in EMOTION_LABELS_EN:
                skipped["unknown_emotion"] += 1
                continue
            emotion = EMOTION_LABELS_EN[emotion_zh]
            video_id = row.get("video_id")
            user_id = row.get("user_id")

            group_tiers = ["overall"]
            if tier:
                group_tiers.append(tier)
            for group_tier in group_tiers:
                counts[media_type][group_tier][emotion] += 1
                users[media_type][group_tier].add(user_id)
                videos[media_type][group_tier].add(video_id)
                monthly_counts[(media_type, group_tier, month)][emotion] += 1

    summary_rows = []
    for media_type in MEDIA_TYPE_ORDER:
        for tier in TIER_ORDER:
            total = sum(counts[media_type][tier].values())
            if not total:
                continue
            view_count = sum_video_metric(videos[media_type][tier], video_stats, "view_count")
            video_comment_count = sum_video_metric(videos[media_type][tier], video_stats, "video_comment_count")
            top_emotion = max(EMOTION_ORDER, key=lambda emotion: counts[media_type][tier].get(emotion, 0))
            row = {
                "media_type": media_type,
                "media_type_label": MEDIA_TYPE_LABELS[media_type],
                "tier": tier,
                "tier_label": TIER_LABELS[tier],
                "comment_count": total,
                "unique_user_count": len(users[media_type][tier]),
                "video_count": len(videos[media_type][tier]),
                "video_view_count": view_count,
                "video_comment_count": video_comment_count,
                "avg_views_per_video": f"{safe_ratio(view_count, len(videos[media_type][tier])):.3f}",
                "avg_video_comments_per_video": f"{safe_ratio(video_comment_count, len(videos[media_type][tier])):.3f}",
                "analyzed_comments_per_1000_views": f"{per_1000(total, view_count):.6f}",
                "analyzed_comments_per_1000_video_comments": f"{per_1000(total, video_comment_count):.6f}",
                "top_emotion": top_emotion,
            }
            for emotion in EMOTION_ORDER:
                count = counts[media_type][tier].get(emotion, 0)
                row[f"{emotion}_count"] = count
                row[f"{emotion}_ratio"] = f"{safe_ratio(count, total):.6f}"
                row[f"{emotion}_per_1000_views"] = f"{per_1000(count, view_count):.6f}"
                row[f"{emotion}_per_1000_video_comments"] = f"{per_1000(count, video_comment_count):.6f}"
            summary_rows.append(row)

    comparison_rows = []
    for tier in TIER_ORDER:
        traditional_total = sum(counts["traditional"][tier].values())
        emerging_total = sum(counts["emerging"][tier].values())
        if not traditional_total or not emerging_total:
            continue
        traditional_view_count = sum_video_metric(videos["traditional"][tier], video_stats, "view_count")
        emerging_view_count = sum_video_metric(videos["emerging"][tier], video_stats, "view_count")
        traditional_video_comment_count = sum_video_metric(videos["traditional"][tier], video_stats, "video_comment_count")
        emerging_video_comment_count = sum_video_metric(videos["emerging"][tier], video_stats, "video_comment_count")
        for emotion in EMOTION_ORDER:
            traditional_share = safe_ratio(counts["traditional"][tier][emotion], traditional_total)
            emerging_share = safe_ratio(counts["emerging"][tier][emotion], emerging_total)
            traditional_per_1000_views = per_1000(counts["traditional"][tier][emotion], traditional_view_count)
            emerging_per_1000_views = per_1000(counts["emerging"][tier][emotion], emerging_view_count)
            traditional_per_1000_video_comments = per_1000(
                counts["traditional"][tier][emotion],
                traditional_video_comment_count,
            )
            emerging_per_1000_video_comments = per_1000(
                counts["emerging"][tier][emotion],
                emerging_video_comment_count,
            )
            comparison_rows.append(
                {
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "emotion": emotion,
                    "traditional_video_count": len(videos["traditional"][tier]),
                    "emerging_video_count": len(videos["emerging"][tier]),
                    "traditional_video_view_count": traditional_view_count,
                    "emerging_video_view_count": emerging_view_count,
                    "traditional_video_comment_count": traditional_video_comment_count,
                    "emerging_video_comment_count": emerging_video_comment_count,
                    "traditional_ratio": f"{traditional_share:.6f}",
                    "emerging_ratio": f"{emerging_share:.6f}",
                    "difference_traditional_minus_emerging": f"{traditional_share - emerging_share:.6f}",
                    "traditional_emotion_per_1000_views": f"{traditional_per_1000_views:.6f}",
                    "emerging_emotion_per_1000_views": f"{emerging_per_1000_views:.6f}",
                    "difference_emotion_per_1000_views": f"{traditional_per_1000_views - emerging_per_1000_views:.6f}",
                    "log2_ratio_emotion_per_1000_views": f"{log2_ratio(traditional_per_1000_views, emerging_per_1000_views):.6f}",
                    "traditional_emotion_per_1000_video_comments": f"{traditional_per_1000_video_comments:.6f}",
                    "emerging_emotion_per_1000_video_comments": f"{emerging_per_1000_video_comments:.6f}",
                    "difference_emotion_per_1000_video_comments": f"{traditional_per_1000_video_comments - emerging_per_1000_video_comments:.6f}",
                    "log2_ratio_emotion_per_1000_video_comments": f"{log2_ratio(traditional_per_1000_video_comments, emerging_per_1000_video_comments):.6f}",
                    "traditional_count": counts["traditional"][tier][emotion],
                    "emerging_count": counts["emerging"][tier][emotion],
                }
            )

    monthly_rows = []
    for (media_type, tier, month), counter in sorted(monthly_counts.items(), key=lambda item: (item[0][2], item[0][0], TIER_ORDER.index(item[0][1]))):
        total = sum(counter.values())
        row = {
            "month": month,
            "media_type": media_type,
            "media_type_label": MEDIA_TYPE_LABELS[media_type],
            "tier": tier,
            "tier_label": TIER_LABELS[tier],
            "comment_count": total,
        }
        for emotion in EMOTION_ORDER:
            count = counter.get(emotion, 0)
            row[f"{emotion}_count"] = count
            row[f"{emotion}_ratio"] = f"{safe_ratio(count, total):.6f}"
        monthly_rows.append(row)

    summary_fields = [
        "media_type", "media_type_label", "tier", "tier_label", "comment_count",
        "unique_user_count", "video_count", "video_view_count", "video_comment_count",
        "avg_views_per_video", "avg_video_comments_per_video",
        "analyzed_comments_per_1000_views", "analyzed_comments_per_1000_video_comments",
        "top_emotion",
    ]
    for emotion in EMOTION_ORDER:
        summary_fields.extend(
            [
                f"{emotion}_count",
                f"{emotion}_ratio",
                f"{emotion}_per_1000_views",
                f"{emotion}_per_1000_video_comments",
            ]
        )
    comparison_fields = [
        "tier", "tier_label", "emotion",
        "traditional_video_count", "emerging_video_count",
        "traditional_video_view_count", "emerging_video_view_count",
        "traditional_video_comment_count", "emerging_video_comment_count",
        "traditional_ratio", "emerging_ratio", "difference_traditional_minus_emerging",
        "traditional_emotion_per_1000_views", "emerging_emotion_per_1000_views",
        "difference_emotion_per_1000_views", "log2_ratio_emotion_per_1000_views",
        "traditional_emotion_per_1000_video_comments", "emerging_emotion_per_1000_video_comments",
        "difference_emotion_per_1000_video_comments", "log2_ratio_emotion_per_1000_video_comments",
        "traditional_count", "emerging_count",
    ]
    monthly_fields = ["month", "media_type", "media_type_label", "tier", "tier_label", "comment_count"]
    for emotion in EMOTION_ORDER:
        monthly_fields.extend([f"{emotion}_count", f"{emotion}_ratio"])

    write_csv(output_dir / "media_type_emotion_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "traditional_vs_emerging_emotion_comparison.csv", comparison_rows, comparison_fields)
    write_csv(output_dir / "media_type_monthly_emotion_summary.csv", monthly_rows, monthly_fields)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analyze_emotion_by_media_type.py",
        "statistics": {
            "comments_jsonl": str(args.comments_jsonl),
            "commenters_json": str(args.commenters_json),
            "videos_json": str(args.videos_json),
            "total_emotion_records": total_records,
            "included_records": sum(sum(counts[media_type]["overall"].values()) for media_type in MEDIA_TYPE_ORDER),
            "max_month": args.max_month,
            "excluded_months": sorted(excluded_months),
            "skipped": dict(skipped),
            "media_video_counts": dict(media_video_counts),
            "media_view_counts": dict(media_view_counts),
            "media_video_comment_counts": dict(media_video_comment_counts),
            "channel_video_counts": dict(channel_video_counts),
        },
        "emotion_order": EMOTION_ORDER,
        "tier_order": TIER_ORDER,
        "outputs": {
            "summary_csv": str(output_dir / "media_type_emotion_summary.csv"),
            "comparison_csv": str(output_dir / "traditional_vs_emerging_emotion_comparison.csv"),
            "monthly_csv": str(output_dir / "media_type_monthly_emotion_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'media_type_emotion_summary.csv'}")
    print(f"Wrote {output_dir / 'traditional_vs_emerging_emotion_comparison.csv'}")
    print(f"Wrote {output_dir / 'media_type_monthly_emotion_summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
