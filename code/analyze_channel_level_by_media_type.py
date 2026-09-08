#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_emotion_by_media_type import EMOTION_LABELS_EN, EMOTION_ORDER
from analyze_topic_by_media_type import TOPICS, normalize_text, topic_hits
from channel_media_types import MEDIA_TYPE_LABELS, display_label_for_channel, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
TOPIC_OUTPUT_DIR = BASE_DIR / "topic_by_media_type"
EMOTION_OUTPUT_DIR = BASE_DIR / "emotion_by_media_type"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
EXCLUDED_MONTHS = {"2025-11"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate topic and emotion responses at the channel level."
    )
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--topic-output-dir", default=TOPIC_OUTPUT_DIR)
    parser.add_argument("--emotion-output-dir", default=EMOTION_OUTPUT_DIR)
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
    tiers = {}
    for row in load_json_data(path):
        user_id = row.get("authorChannelId")
        if not user_id:
            continue
        tier = assign_tier(int(row.get("comment_count", 0) or 0))
        if tier:
            tiers[user_id] = tier
    return tiers


def parse_count(value) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_video_metadata(path: Path):
    video_channel = {}
    video_media_type = {}
    video_stats = {}
    for video in load_json_data(path):
        video_id = video.get("id")
        if not video_id:
            continue
        snippet = video.get("snippet") or {}
        statistics = video.get("statistics") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        media_type = media_type_for_channel(channel)
        video_channel[video_id] = channel
        video_media_type[video_id] = media_type
        video_stats[video_id] = {
            "view_count": parse_count(statistics.get("viewCount")),
            "video_comment_count": parse_count(statistics.get("commentCount")),
        }
    return video_channel, video_media_type, video_stats


def init_channel_stats():
    return {
        "comments": 0,
        "users": set(),
        "videos": set(),
        "comments_with_topic": 0,
        "topic_comment_counts": Counter(),
        "topic_mention_counts": Counter(),
        "emotion_counts": Counter(),
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def per_1000(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator * 1000 if denominator else 0.0


def sum_video_metric(video_ids: set[str], video_stats: dict[str, dict], metric: str) -> int:
    return sum(video_stats.get(video_id, {}).get(metric, 0) for video_id in video_ids)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def direction(value: float) -> str:
    if value > 0:
        return "traditional_higher"
    if value < 0:
        return "emerging_higher"
    return "same"


def channel_base(channel: str, media_type: str, stat: dict, video_stats: dict[str, dict]) -> dict:
    view_count = sum_video_metric(stat["videos"], video_stats, "view_count")
    video_comment_count = sum_video_metric(stat["videos"], video_stats, "video_comment_count")
    return {
        "channel": channel,
        "channel_label": display_label_for_channel(channel),
        "media_type": media_type,
        "media_type_label": MEDIA_TYPE_LABELS[media_type],
        "comment_count": stat["comments"],
        "unique_user_count": len(stat["users"]),
        "video_count": len(stat["videos"]),
        "video_view_count": view_count,
        "video_comment_count": video_comment_count,
        "avg_views_per_video": f"{safe_ratio(view_count, len(stat['videos'])):.3f}",
        "avg_video_comments_per_video": f"{safe_ratio(video_comment_count, len(stat['videos'])):.3f}",
        "analyzed_comments_per_1000_views": f"{per_1000(stat['comments'], view_count):.6f}",
        "analyzed_comments_per_1000_video_comments": f"{per_1000(stat['comments'], video_comment_count):.6f}",
    }


def build_support_rows(rows: list[dict], item_key: str, count_key: str, share_key: str, per_view_key: str) -> list[dict]:
    items = sorted({row[item_key] for row in rows})
    output = []
    for item in items:
        item_rows = [row for row in rows if row[item_key] == item]
        by_media = {
            media_type: [row for row in item_rows if row["media_type"] == media_type]
            for media_type in MEDIA_TYPE_ORDER
        }
        if not all(by_media.values()):
            continue

        def mean(media_type: str, key: str) -> float:
            values = [float(row[key]) for row in by_media[media_type]]
            return sum(values) / len(values) if values else 0.0

        weighted = {}
        for media_type in MEDIA_TYPE_ORDER:
            total_count = sum(int(row[count_key]) for row in by_media[media_type])
            total_comments = sum(int(row["comment_count"]) for row in by_media[media_type])
            total_views = sum(int(row["video_view_count"]) for row in by_media[media_type])
            weighted[media_type] = {
                "share": safe_ratio(total_count, total_comments),
                "per_1000_views": per_1000(total_count, total_views),
            }

        share_diff = mean("traditional", share_key) - mean("emerging", share_key)
        intensity_diff = mean("traditional", per_view_key) - mean("emerging", per_view_key)
        weighted_share_diff = weighted["traditional"]["share"] - weighted["emerging"]["share"]
        weighted_intensity_diff = weighted["traditional"]["per_1000_views"] - weighted["emerging"]["per_1000_views"]
        output.append(
            {
                item_key: item,
                "traditional_channel_count": len(by_media["traditional"]),
                "emerging_channel_count": len(by_media["emerging"]),
                "traditional_channel_mean_share": f"{mean('traditional', share_key):.6f}",
                "emerging_channel_mean_share": f"{mean('emerging', share_key):.6f}",
                "difference_channel_mean_share": f"{share_diff:.6f}",
                "traditional_weighted_share": f"{weighted['traditional']['share']:.6f}",
                "emerging_weighted_share": f"{weighted['emerging']['share']:.6f}",
                "difference_weighted_share": f"{weighted_share_diff:.6f}",
                "traditional_channel_mean_per_1000_views": f"{mean('traditional', per_view_key):.6f}",
                "emerging_channel_mean_per_1000_views": f"{mean('emerging', per_view_key):.6f}",
                "difference_channel_mean_per_1000_views": f"{intensity_diff:.6f}",
                "traditional_weighted_per_1000_views": f"{weighted['traditional']['per_1000_views']:.6f}",
                "emerging_weighted_per_1000_views": f"{weighted['emerging']['per_1000_views']:.6f}",
                "difference_weighted_per_1000_views": f"{weighted_intensity_diff:.6f}",
                "channel_mean_share_direction": direction(share_diff),
                "weighted_share_direction": direction(weighted_share_diff),
                "channel_mean_intensity_direction": direction(intensity_diff),
                "weighted_intensity_direction": direction(weighted_intensity_diff),
                "share_direction_matches_weighted": direction(share_diff) == direction(weighted_share_diff),
                "intensity_direction_matches_weighted": direction(intensity_diff) == direction(weighted_intensity_diff),
            }
        )
    output.sort(key=lambda row: float(row["difference_channel_mean_per_1000_views"]))
    return output


def main() -> None:
    args = parse_args()
    topic_output_dir = Path(args.topic_output_dir)
    emotion_output_dir = Path(args.emotion_output_dir)
    topic_output_dir.mkdir(parents=True, exist_ok=True)
    emotion_output_dir.mkdir(parents=True, exist_ok=True)
    excluded_months = set(args.exclude_month or [])

    video_channel, video_media_type, video_stats = load_video_metadata(Path(args.videos_json))

    channel_stats = defaultdict(init_channel_stats)
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
            video_id = row.get("video_id")
            channel = video_channel.get(video_id, "")
            media_type = video_media_type.get(video_id, "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                skipped["unknown_media_type"] += 1
                continue

            emotion_zh = row.get("emotion_label")
            if emotion_zh not in EMOTION_LABELS_EN:
                skipped["unknown_emotion"] += 1
                continue

            stat = channel_stats[(channel, media_type)]
            stat["comments"] += 1
            stat["users"].add(row.get("user_id"))
            stat["videos"].add(video_id)
            stat["emotion_counts"][EMOTION_LABELS_EN[emotion_zh]] += 1

            text = normalize_text(row.get("text", ""))
            if not text:
                skipped["empty_text_for_topic"] += 1
                continue
            hits, _ = topic_hits(text)
            if hits:
                stat["comments_with_topic"] += 1
            for topic, mention_count in hits.items():
                stat["topic_comment_counts"][topic] += 1
                stat["topic_mention_counts"][topic] += mention_count

    topic_rows = []
    emotion_rows = []
    for (channel, media_type), stat in sorted(
        channel_stats.items(),
        key=lambda item: (MEDIA_TYPE_ORDER.index(item[0][1]), item[0][0]),
    ):
        if not stat["comments"]:
            continue
        base = channel_base(channel, media_type, stat, video_stats)
        view_count = int(base["video_view_count"])
        video_comment_count = int(base["video_comment_count"])
        topic_mentions_total = sum(stat["topic_mention_counts"].values())
        for topic in TOPICS:
            topic_comment_count = stat["topic_comment_counts"][topic]
            topic_mention_count = stat["topic_mention_counts"][topic]
            topic_rows.append(
                {
                    **base,
                    "comments_with_any_topic": stat["comments_with_topic"],
                    "any_topic_comment_share": f"{safe_ratio(stat['comments_with_topic'], stat['comments']):.6f}",
                    "any_topic_comments_per_1000_views": f"{per_1000(stat['comments_with_topic'], view_count):.6f}",
                    "any_topic_comments_per_1000_video_comments": f"{per_1000(stat['comments_with_topic'], video_comment_count):.6f}",
                    "topic": topic,
                    "topic_comment_count": topic_comment_count,
                    "topic_comment_share": f"{safe_ratio(topic_comment_count, stat['comments']):.6f}",
                    "topic_comments_per_1000_views": f"{per_1000(topic_comment_count, view_count):.6f}",
                    "topic_comments_per_1000_video_comments": f"{per_1000(topic_comment_count, video_comment_count):.6f}",
                    "topic_mention_count": topic_mention_count,
                    "topic_mention_share": f"{safe_ratio(topic_mention_count, topic_mentions_total):.6f}",
                    "topic_mentions_per_1000_views": f"{per_1000(topic_mention_count, view_count):.6f}",
                    "topic_mentions_per_1000_video_comments": f"{per_1000(topic_mention_count, video_comment_count):.6f}",
                }
            )
        for emotion in EMOTION_ORDER:
            emotion_count = stat["emotion_counts"][emotion]
            emotion_rows.append(
                {
                    **base,
                    "emotion": emotion,
                    "emotion_count": emotion_count,
                    "emotion_share": f"{safe_ratio(emotion_count, stat['comments']):.6f}",
                    "emotion_per_1000_views": f"{per_1000(emotion_count, view_count):.6f}",
                    "emotion_per_1000_video_comments": f"{per_1000(emotion_count, video_comment_count):.6f}",
                }
            )

    topic_support_rows = build_support_rows(
        topic_rows,
        item_key="topic",
        count_key="topic_comment_count",
        share_key="topic_comment_share",
        per_view_key="topic_comments_per_1000_views",
    )
    emotion_support_rows = build_support_rows(
        emotion_rows,
        item_key="emotion",
        count_key="emotion_count",
        share_key="emotion_share",
        per_view_key="emotion_per_1000_views",
    )

    base_fields = [
        "channel", "channel_label", "media_type", "media_type_label", "comment_count", "unique_user_count",
        "video_count", "video_view_count", "video_comment_count",
        "avg_views_per_video", "avg_video_comments_per_video",
        "analyzed_comments_per_1000_views", "analyzed_comments_per_1000_video_comments",
    ]
    topic_fields = base_fields + [
        "comments_with_any_topic", "any_topic_comment_share",
        "any_topic_comments_per_1000_views", "any_topic_comments_per_1000_video_comments",
        "topic", "topic_comment_count", "topic_comment_share",
        "topic_comments_per_1000_views", "topic_comments_per_1000_video_comments",
        "topic_mention_count", "topic_mention_share",
        "topic_mentions_per_1000_views", "topic_mentions_per_1000_video_comments",
    ]
    emotion_fields = base_fields + [
        "emotion", "emotion_count", "emotion_share",
        "emotion_per_1000_views", "emotion_per_1000_video_comments",
    ]
    support_fields = [
        "traditional_channel_count", "emerging_channel_count",
        "traditional_channel_mean_share", "emerging_channel_mean_share",
        "difference_channel_mean_share", "traditional_weighted_share",
        "emerging_weighted_share", "difference_weighted_share",
        "traditional_channel_mean_per_1000_views", "emerging_channel_mean_per_1000_views",
        "difference_channel_mean_per_1000_views",
        "traditional_weighted_per_1000_views", "emerging_weighted_per_1000_views",
        "difference_weighted_per_1000_views",
        "channel_mean_share_direction", "weighted_share_direction",
        "channel_mean_intensity_direction", "weighted_intensity_direction",
        "share_direction_matches_weighted", "intensity_direction_matches_weighted",
    ]

    write_csv(topic_output_dir / "channel_topic_summary.csv", topic_rows, topic_fields)
    write_csv(emotion_output_dir / "channel_emotion_summary.csv", emotion_rows, emotion_fields)
    write_csv(
        topic_output_dir / "channel_topic_media_type_support.csv",
        topic_support_rows,
        ["topic"] + support_fields,
    )
    write_csv(
        emotion_output_dir / "channel_emotion_media_type_support.csv",
        emotion_support_rows,
        ["emotion"] + support_fields,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": Path(__file__).name,
        "comments_jsonl": str(args.comments_jsonl),
        "commenters_json": str(args.commenters_json),
        "videos_json": str(args.videos_json),
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "total_records": total_records,
        "included_comments": sum(stat["comments"] for stat in channel_stats.values()),
        "channel_count_by_media_type": {
            media_type: sum(1 for (_, mt), stat in channel_stats.items() if mt == media_type and stat["comments"])
            for media_type in MEDIA_TYPE_ORDER
        },
        "skipped": dict(skipped),
        "outputs": {
            "topic_channel_summary_csv": str(topic_output_dir / "channel_topic_summary.csv"),
            "emotion_channel_summary_csv": str(emotion_output_dir / "channel_emotion_summary.csv"),
            "topic_support_csv": str(topic_output_dir / "channel_topic_media_type_support.csv"),
            "emotion_support_csv": str(emotion_output_dir / "channel_emotion_media_type_support.csv"),
        },
    }
    (topic_output_dir / "channel_level_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (emotion_output_dir / "channel_level_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {topic_output_dir / 'channel_topic_summary.csv'}")
    print(f"Wrote {emotion_output_dir / 'channel_emotion_summary.csv'}")
    print(f"Wrote {topic_output_dir / 'channel_topic_media_type_support.csv'}")
    print(f"Wrote {emotion_output_dir / 'channel_emotion_media_type_support.csv'}")
    print(f"Wrote {topic_output_dir / 'channel_level_summary.json'}")
    print(f"Wrote {emotion_output_dir / 'channel_level_summary.json'}")


if __name__ == "__main__":
    main()
