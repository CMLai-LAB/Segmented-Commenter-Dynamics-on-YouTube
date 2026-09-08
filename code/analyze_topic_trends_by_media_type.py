#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_topic_by_media_type import TOPICS, load_json_data
from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "topic_by_media_type"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
TIER_ORDER = ["overall", "low_2_5", "mid_6_20", "core_21_50", "high_51_plus"]
TIER_LABELS = {
    "overall": "Overall (2+ comments)",
    "low_2_5": "2-5 comments",
    "mid_6_20": "6-20 comments",
    "core_21_50": "21-50 comments",
    "high_51_plus": "51+ comments",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze quarterly topic-share trends by media type.")
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-month", default="2025-10")
    parser.add_argument("--exclude-month", action="append", default=["2025-11"])
    parser.add_argument(
        "--include-partial-quarters",
        action="store_true",
        help="Include quarters with fewer than three observed months, e.g. 2025-Q4 when data ends at 2025-10.",
    )
    return parser.parse_args()


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
    tiers = {}
    for row in load_json_data(path):
        user_id = row.get("authorChannelId") or row.get("user_id")
        if not user_id:
            continue
        tier = assign_tier(int(row.get("comment_count", 0) or 0))
        if tier:
            tiers[user_id] = tier
    return tiers


def load_video_media_types(path: Path) -> dict[str, str]:
    mapping = {}
    for video in load_json_data(path):
        video_id = video.get("id") or video.get("video_id")
        snippet = video.get("snippet") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        if video_id:
            mapping[video_id] = media_type_for_channel(channel)
    return mapping


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def topic_hits(text: str) -> set[str]:
    hits = set()
    for topic, keywords in TOPICS.items():
        if any(keyword in text for keyword in keywords):
            hits.add(topic)
    return hits


def month_to_quarter(month: str) -> str:
    year, month_num = month.split("-")
    quarter = (int(month_num) - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def pct(value: int | float, denom: int | float) -> float:
    return value / denom if denom else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_months = set(args.exclude_month or [])

    user_tiers = load_user_tiers(Path(args.commenters_json))
    video_media_types = load_video_media_types(Path(args.videos_json))

    comments = Counter()
    comments_with_topic = Counter()
    topic_counts = defaultdict(Counter)
    quarter_months = defaultdict(set)
    skipped = Counter()

    with Path(args.comments_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            month = row.get("month") or ""
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
            text = normalize_text(row.get("text", ""))
            if not text:
                skipped["empty_text"] += 1
                continue
            quarter = month_to_quarter(month)
            quarter_months[quarter].add(month)
            hits = topic_hits(text)
            for group_tier in ("overall", tier):
                key = (quarter, media_type, group_tier)
                comments[key] += 1
                if hits:
                    comments_with_topic[key] += 1
                for topic in hits:
                    topic_counts[key][topic] += 1

    all_quarters = sorted({key[0] for key in comments})
    partial_quarters = {
        quarter: sorted(months)
        for quarter, months in sorted(quarter_months.items())
        if len(months) < 3
    }
    quarters = all_quarters if args.include_partial_quarters else [
        quarter for quarter in all_quarters if quarter not in partial_quarters
    ]

    rows = []
    for quarter in quarters:
        for media_type in MEDIA_TYPE_ORDER:
            for tier in TIER_ORDER:
                key = (quarter, media_type, tier)
                total = comments[key]
                for topic in TOPICS:
                    count = topic_counts[key][topic]
                    rows.append(
                        {
                            "quarter": quarter,
                            "media_type": media_type,
                            "media_type_label": MEDIA_TYPE_LABELS[media_type],
                            "tier": tier,
                            "tier_label": TIER_LABELS[tier],
                            "comment_count": total,
                            "comments_with_any_topic": comments_with_topic[key],
                            "any_topic_comment_share": f"{pct(comments_with_topic[key], total):.6f}",
                            "topic": topic,
                            "topic_comment_count": count,
                            "topic_comment_share": f"{pct(count, total):.6f}",
                        }
                    )

    output_csv = output_dir / "media_type_quarter_topic_summary.csv"
    output_json = output_dir / "media_type_quarter_topic_summary.json"
    write_csv(
        output_csv,
        rows,
        [
            "quarter", "media_type", "media_type_label", "tier", "tier_label",
            "comment_count", "comments_with_any_topic", "any_topic_comment_share",
            "topic", "topic_comment_count", "topic_comment_share",
        ],
    )
    output_json.write_text(
        json.dumps(
            {
                "generated_by": Path(__file__).name,
                "input_comments_jsonl": str(args.comments_jsonl),
                "max_month": args.max_month,
                "include_partial_quarters": args.include_partial_quarters,
                "quarters": quarters,
                "excluded_partial_quarters": {} if args.include_partial_quarters else partial_quarters,
                "skipped": dict(skipped),
                "output_csv": str(output_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
