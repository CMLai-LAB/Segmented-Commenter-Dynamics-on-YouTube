#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from channel_media_types import MEDIA_TYPE_LABELS, media_type_for_channel


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
COMMENTERS_JSON = BASE_DIR / "all_unique_commenters.json"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "topic_by_media_type"
EXCLUDED_MONTHS = {"2025-11"}

TIER_ORDER = ["overall", "low_2_5", "mid_6_20", "core_21_50", "high_51_plus"]
TIER_LABELS = {
    "overall": "Overall (all comments)",
    "low_2_5": "2-5 comments",
    "mid_6_20": "6-20 comments",
    "core_21_50": "21-50 comments",
    "high_51_plus": "51+ comments",
}
MEDIA_TYPE_ORDER = ["traditional", "emerging"]

TOPICS = {
    "政治行為與事件": [
        "罷免", "連署", "投票", "選舉", "罷掉", "站出來", "選票",
        "同意", "罷團", "青鳥", "倒閣", "公投", "抗議", "示威",
    ],
    "政黨與政治陣營": [
        "民進黨", "國民黨", "民眾黨", "藍白", "執政黨", "在野黨",
        "藍營", "綠營", "政黨", "共產黨", "政治陣營",
    ],
    "政治制度與價值": [
        "民主", "自由", "公民", "獨裁", "司法", "言論", "正義",
        "監督", "民意", "制度", "歷史", "憲法", "法治", "人權",
    ],
    "國家與地緣政治": [
        "中國", "中共", "美國", "日本", "香港", "台灣", "反共",
        "統一", "對岸", "戰爭", "中國人", "地緣", "主權",
    ],
    "政治人物": [
        "賴清德", "柯文哲", "黃國昌", "韓國瑜", "朱立倫",
        "徐巧芯", "館長", "柯建銘", "政治人物", "候選人",
    ],
    "情緒評價與社群語言": [
        "加油", "笑死", "謝謝", "辛苦", "可悲", "噁心", "無恥",
        "悲哀", "造謠", "洗腦", "笑話", "諷刺", "批評", "嘲諷",
    ],
    "政治職位與機構": [
        "立法院", "立委", "國會", "院長", "委員", "議員",
        "主席", "黨主席", "市長", "總統", "行政院", "政府",
    ],
    "政策與治理議題": [
        "預算", "補助", "關稅", "經濟", "法案", "政策",
        "貪污", "詐騙", "浪費", "違法", "改革", "施政",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze recall-comment topic distribution by media type and commenter tier.")
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--exclude-month", action="append", default=sorted(EXCLUDED_MONTHS))
    parser.add_argument("--max-month", default="2025-10")
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


def load_json_data(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


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
    video_media_type = {}
    video_channel = {}
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
        video_channel[video_id] = channel
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
        video_channel,
        video_stats,
        channel_counts,
        media_counts,
        media_view_counts,
        media_video_comment_counts,
    )


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_occurrences(text: str, keyword: str) -> int:
    return text.count(keyword)


def topic_hits(text: str):
    hits = {}
    keyword_hits = {}
    for topic, keywords in TOPICS.items():
        counts = Counter()
        for keyword in keywords:
            count = count_occurrences(text, keyword)
            if count:
                counts[keyword] = count
        if counts:
            hits[topic] = sum(counts.values())
            keyword_hits[topic] = counts
    return hits, keyword_hits


def init_group_stats():
    return {
        "comments": 0,
        "users": set(),
        "videos": set(),
        "comments_with_topic": 0,
        "topic_comment_counts": Counter(),
        "topic_mention_counts": Counter(),
        "keyword_counts": defaultdict(Counter),
    }


def pct(value: int | float, denom: int | float) -> float:
    return value / denom if denom else 0.0


def per_1000(value: int | float, denom: int | float) -> float:
    return value / denom * 1000 if denom else 0.0


def log2_ratio(a_share: float, b_share: float, smoothing: float = 1e-9) -> float:
    return math.log2((a_share + smoothing) / (b_share + smoothing))


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

    user_tiers = load_user_tiers(Path(args.commenters_json))
    (
        video_media_type,
        video_channel,
        video_stats,
        channel_counts,
        media_video_counts,
        media_view_counts,
        media_video_comment_counts,
    ) = load_video_metadata(Path(args.videos_json))

    groups = defaultdict(init_group_stats)
    unknown_videos = Counter()
    unknown_channels = Counter()
    skipped = Counter()

    with Path(args.comments_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            month = row.get("month")
            if month and month > args.max_month:
                skipped["after_max_month"] += 1
                continue
            if row.get("month") in excluded_months:
                skipped["excluded_month"] += 1
                continue
            tier = user_tiers.get(row.get("user_id"))
            if not tier:
                skipped["one_comment_or_missing_user_tier_excluded_from_tiers"] += 1
            video_id = row.get("video_id")
            media_type = video_media_type.get(video_id, "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                unknown_videos[video_id or ""] += 1
                unknown_channels[video_channel.get(video_id, "")] += 1
                skipped["unknown_media_type"] += 1
                continue

            text = normalize_text(row.get("text", ""))
            if not text:
                skipped["empty_text"] += 1
                continue

            hits, keyword_hits = topic_hits(text)
            keys = [(media_type, "overall")]
            if tier:
                keys.append((media_type, tier))
            for key in keys:
                stat = groups[key]
                stat["comments"] += 1
                stat["users"].add(row.get("user_id"))
                stat["videos"].add(video_id)
                if hits:
                    stat["comments_with_topic"] += 1
                for topic, mention_count in hits.items():
                    stat["topic_comment_counts"][topic] += 1
                    stat["topic_mention_counts"][topic] += mention_count
                    stat["keyword_counts"][topic].update(keyword_hits[topic])

    summary_rows = []
    keyword_rows = []
    for media_type in MEDIA_TYPE_ORDER:
        for tier in TIER_ORDER:
            stat = groups.get((media_type, tier))
            if not stat:
                continue
            comments = stat["comments"]
            view_count = sum_video_metric(stat["videos"], video_stats, "view_count")
            video_comment_count = sum_video_metric(stat["videos"], video_stats, "video_comment_count")
            topic_mentions_total = sum(stat["topic_mention_counts"].values())
            for topic in TOPICS:
                topic_comments = stat["topic_comment_counts"][topic]
                topic_mentions = stat["topic_mention_counts"][topic]
                top_keywords = stat["keyword_counts"][topic].most_common(8)
                summary_rows.append(
                    {
                        "media_type": media_type,
                        "media_type_label": MEDIA_TYPE_LABELS[media_type],
                        "tier": tier,
                        "tier_label": TIER_LABELS[tier],
                        "comment_count": comments,
                        "unique_user_count": len(stat["users"]),
                        "video_count": len(stat["videos"]),
                        "video_view_count": view_count,
                        "video_comment_count": video_comment_count,
                        "avg_views_per_video": f"{pct(view_count, len(stat['videos'])):.3f}",
                        "avg_video_comments_per_video": f"{pct(video_comment_count, len(stat['videos'])):.3f}",
                        "analyzed_comments_per_1000_views": f"{per_1000(comments, view_count):.6f}",
                        "analyzed_comments_per_1000_video_comments": f"{per_1000(comments, video_comment_count):.6f}",
                        "comments_with_any_topic": stat["comments_with_topic"],
                        "any_topic_comment_share": f"{pct(stat['comments_with_topic'], comments):.6f}",
                        "any_topic_comments_per_1000_views": f"{per_1000(stat['comments_with_topic'], view_count):.6f}",
                        "any_topic_comments_per_1000_video_comments": f"{per_1000(stat['comments_with_topic'], video_comment_count):.6f}",
                        "topic": topic,
                        "topic_comment_count": topic_comments,
                        "topic_comment_share": f"{pct(topic_comments, comments):.6f}",
                        "topic_comments_per_1000_views": f"{per_1000(topic_comments, view_count):.6f}",
                        "topic_comments_per_1000_video_comments": f"{per_1000(topic_comments, video_comment_count):.6f}",
                        "topic_mention_count": topic_mentions,
                        "topic_mention_share": f"{pct(topic_mentions, topic_mentions_total):.6f}",
                        "topic_mentions_per_1000_views": f"{per_1000(topic_mentions, view_count):.6f}",
                        "topic_mentions_per_1000_video_comments": f"{per_1000(topic_mentions, video_comment_count):.6f}",
                        "top_keywords": "; ".join(f"{kw}:{count}" for kw, count in top_keywords),
                    }
                )
                for keyword, count in stat["keyword_counts"][topic].most_common():
                    keyword_rows.append(
                        {
                            "media_type": media_type,
                            "media_type_label": MEDIA_TYPE_LABELS[media_type],
                            "tier": tier,
                            "tier_label": TIER_LABELS[tier],
                            "comment_count": comments,
                            "video_count": len(stat["videos"]),
                            "video_view_count": view_count,
                            "video_comment_count": video_comment_count,
                            "topic": topic,
                            "keyword": keyword,
                            "count": count,
                            "share_within_topic_mentions": f"{pct(count, topic_mentions):.6f}",
                            "keyword_mentions_per_1000_views": f"{per_1000(count, view_count):.6f}",
                            "keyword_mentions_per_1000_video_comments": f"{per_1000(count, video_comment_count):.6f}",
                        }
                    )

    comparison_rows = []
    for tier in TIER_ORDER:
        traditional = groups.get(("traditional", tier))
        emerging = groups.get(("emerging", tier))
        if not traditional or not emerging:
            continue
        traditional_view_count = sum_video_metric(traditional["videos"], video_stats, "view_count")
        emerging_view_count = sum_video_metric(emerging["videos"], video_stats, "view_count")
        traditional_video_comment_count = sum_video_metric(traditional["videos"], video_stats, "video_comment_count")
        emerging_video_comment_count = sum_video_metric(emerging["videos"], video_stats, "video_comment_count")
        for topic in TOPICS:
            traditional_share = pct(traditional["topic_comment_counts"][topic], traditional["comments"])
            emerging_share = pct(emerging["topic_comment_counts"][topic], emerging["comments"])
            traditional_per_1000_views = per_1000(traditional["topic_comment_counts"][topic], traditional_view_count)
            emerging_per_1000_views = per_1000(emerging["topic_comment_counts"][topic], emerging_view_count)
            traditional_per_1000_video_comments = per_1000(
                traditional["topic_comment_counts"][topic],
                traditional_video_comment_count,
            )
            emerging_per_1000_video_comments = per_1000(
                emerging["topic_comment_counts"][topic],
                emerging_video_comment_count,
            )
            comparison_rows.append(
                {
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "topic": topic,
                    "traditional_video_count": len(traditional["videos"]),
                    "emerging_video_count": len(emerging["videos"]),
                    "traditional_video_view_count": traditional_view_count,
                    "emerging_video_view_count": emerging_view_count,
                    "traditional_video_comment_count": traditional_video_comment_count,
                    "emerging_video_comment_count": emerging_video_comment_count,
                    "traditional_comment_share": f"{traditional_share:.6f}",
                    "emerging_comment_share": f"{emerging_share:.6f}",
                    "difference_traditional_minus_emerging": f"{traditional_share - emerging_share:.6f}",
                    "log2_ratio_traditional_vs_emerging": f"{log2_ratio(traditional_share, emerging_share):.6f}",
                    "traditional_topic_comments_per_1000_views": f"{traditional_per_1000_views:.6f}",
                    "emerging_topic_comments_per_1000_views": f"{emerging_per_1000_views:.6f}",
                    "difference_topic_comments_per_1000_views": f"{traditional_per_1000_views - emerging_per_1000_views:.6f}",
                    "log2_ratio_topic_comments_per_1000_views": f"{log2_ratio(traditional_per_1000_views, emerging_per_1000_views):.6f}",
                    "traditional_topic_comments_per_1000_video_comments": f"{traditional_per_1000_video_comments:.6f}",
                    "emerging_topic_comments_per_1000_video_comments": f"{emerging_per_1000_video_comments:.6f}",
                    "difference_topic_comments_per_1000_video_comments": f"{traditional_per_1000_video_comments - emerging_per_1000_video_comments:.6f}",
                    "log2_ratio_topic_comments_per_1000_video_comments": f"{log2_ratio(traditional_per_1000_video_comments, emerging_per_1000_video_comments):.6f}",
                    "traditional_topic_comment_count": traditional["topic_comment_counts"][topic],
                    "emerging_topic_comment_count": emerging["topic_comment_counts"][topic],
                }
            )

    summary_fields = [
        "media_type", "media_type_label", "tier", "tier_label", "comment_count",
        "unique_user_count", "video_count", "video_view_count", "video_comment_count",
        "avg_views_per_video", "avg_video_comments_per_video",
        "analyzed_comments_per_1000_views", "analyzed_comments_per_1000_video_comments",
        "comments_with_any_topic", "any_topic_comment_share",
        "any_topic_comments_per_1000_views", "any_topic_comments_per_1000_video_comments",
        "topic", "topic_comment_count", "topic_comment_share",
        "topic_comments_per_1000_views", "topic_comments_per_1000_video_comments",
        "topic_mention_count", "topic_mention_share",
        "topic_mentions_per_1000_views", "topic_mentions_per_1000_video_comments",
        "top_keywords",
    ]
    keyword_fields = [
        "media_type", "media_type_label", "tier", "tier_label",
        "comment_count", "video_count", "video_view_count", "video_comment_count",
        "topic", "keyword", "count", "share_within_topic_mentions",
        "keyword_mentions_per_1000_views", "keyword_mentions_per_1000_video_comments",
    ]
    comparison_fields = [
        "tier", "tier_label", "topic",
        "traditional_video_count", "emerging_video_count",
        "traditional_video_view_count", "emerging_video_view_count",
        "traditional_video_comment_count", "emerging_video_comment_count",
        "traditional_comment_share", "emerging_comment_share",
        "difference_traditional_minus_emerging", "log2_ratio_traditional_vs_emerging",
        "traditional_topic_comments_per_1000_views", "emerging_topic_comments_per_1000_views",
        "difference_topic_comments_per_1000_views", "log2_ratio_topic_comments_per_1000_views",
        "traditional_topic_comments_per_1000_video_comments", "emerging_topic_comments_per_1000_video_comments",
        "difference_topic_comments_per_1000_video_comments", "log2_ratio_topic_comments_per_1000_video_comments",
        "traditional_topic_comment_count", "emerging_topic_comment_count",
    ]

    write_csv(output_dir / "media_type_topic_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "media_type_topic_keywords.csv", keyword_rows, keyword_fields)
    write_csv(output_dir / "traditional_vs_emerging_topic_comparison.csv", comparison_rows, comparison_fields)

    group_summary = {}
    for media_type in MEDIA_TYPE_ORDER:
        group_summary[media_type] = {}
        for tier in TIER_ORDER:
            stat = groups.get((media_type, tier))
            if not stat:
                continue
            view_count = sum_video_metric(stat["videos"], video_stats, "view_count")
            video_comment_count = sum_video_metric(stat["videos"], video_stats, "video_comment_count")
            group_summary[media_type][tier] = {
                "label": TIER_LABELS[tier],
                "comment_count": stat["comments"],
                "unique_user_count": len(stat["users"]),
                "video_count": len(stat["videos"]),
                "video_view_count": view_count,
                "video_comment_count": video_comment_count,
                "analyzed_comments_per_1000_views": per_1000(stat["comments"], view_count),
                "analyzed_comments_per_1000_video_comments": per_1000(stat["comments"], video_comment_count),
                "comments_with_any_topic": stat["comments_with_topic"],
                "any_topic_comment_share": pct(stat["comments_with_topic"], stat["comments"]),
                "any_topic_comments_per_1000_views": per_1000(stat["comments_with_topic"], view_count),
                "any_topic_comments_per_1000_video_comments": per_1000(stat["comments_with_topic"], video_comment_count),
                "top_topics_by_comment_share": [
                    {
                        "topic": topic,
                        "comment_count": stat["topic_comment_counts"][topic],
                        "comment_share": pct(stat["topic_comment_counts"][topic], stat["comments"]),
                        "comments_per_1000_views": per_1000(stat["topic_comment_counts"][topic], view_count),
                        "comments_per_1000_video_comments": per_1000(stat["topic_comment_counts"][topic], video_comment_count),
                        "top_keywords": [
                            {"keyword": kw, "count": count}
                            for kw, count in stat["keyword_counts"][topic].most_common(5)
                        ],
                    }
                    for topic in sorted(
                        TOPICS,
                        key=lambda item: stat["topic_comment_counts"][item],
                        reverse=True,
                    )
                ],
            }

    summary = {
        "generated_by": "analyze_topic_by_media_type.py",
        "comments_jsonl": str(args.comments_jsonl),
        "commenters_json": str(args.commenters_json),
        "videos_json": str(args.videos_json),
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "media_video_counts": dict(media_video_counts),
        "media_view_counts": dict(media_view_counts),
        "media_video_comment_counts": dict(media_video_comment_counts),
        "channel_video_counts": dict(channel_counts),
        "skipped": dict(skipped),
        "unknown_channels_by_comment_count": {
            channel: count for channel, count in unknown_channels.most_common()
        },
        "topics": TOPICS,
        "groups": group_summary,
        "outputs": {
            "summary_csv": str(output_dir / "media_type_topic_summary.csv"),
            "keywords_csv": str(output_dir / "media_type_topic_keywords.csv"),
            "comparison_csv": str(output_dir / "traditional_vs_emerging_topic_comparison.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'media_type_topic_summary.csv'}")
    print(f"Wrote {output_dir / 'media_type_topic_keywords.csv'}")
    print(f"Wrote {output_dir / 'traditional_vs_emerging_topic_comparison.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
