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
OUTPUT_DIR = BASE_DIR / "user_mobility_by_media_type"

MEDIA_TYPE_ORDER = ["traditional", "emerging"]
STATE_ORDER = ["traditional_only", "emerging_only", "both"]
CHURN_STATE_ORDER = ["inactive", "traditional_only", "emerging_only", "both"]
STATE_LABELS = {
    "inactive": "Inactive",
    "traditional_only": "Traditional only",
    "emerging_only": "Digital / emerging only",
    "both": "Both media types",
}
EXCLUDED_MONTHS = {"2025-11"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze monthly user mobility between traditional and digital/emerging media."
    )
    parser.add_argument("--comments-jsonl", default=COMMENTS_JSONL)
    parser.add_argument("--commenters-json", default=COMMENTERS_JSON)
    parser.add_argument("--videos-json", default=VIDEOS_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--exclude-month", action="append", default=sorted(EXCLUDED_MONTHS))
    parser.add_argument("--max-month", default="2025-10")
    parser.add_argument(
        "--include-single-commenters",
        action="store_true",
        help="Include users with only one observed comment. By default, mobility follows the 2+ commenter filter used elsewhere.",
    )
    return parser.parse_args()


def load_json_data(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


def load_eligible_users(path: Path, include_single_commenters: bool) -> set[str]:
    users = set()
    for row in load_json_data(path):
        user_id = row.get("authorChannelId") or row.get("user_id")
        if not user_id:
            continue
        comment_count = int(row.get("comment_count", 0) or 0)
        if include_single_commenters or comment_count > 1:
            users.add(user_id)
    return users


def load_video_media_types(path: Path) -> dict[str, str]:
    mapping = {}
    for video in load_json_data(path):
        video_id = video.get("id") or video.get("video_id")
        if not video_id:
            continue
        snippet = video.get("snippet") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        mapping[video_id] = media_type_for_channel(channel)
    return mapping


def state_from_media_set(media_types: set[str]) -> str:
    has_traditional = "traditional" in media_types
    has_emerging = "emerging" in media_types
    if has_traditional and has_emerging:
        return "both"
    if has_traditional:
        return "traditional_only"
    if has_emerging:
        return "emerging_only"
    return "inactive"


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transition_label(from_state: str, to_state: str) -> str:
    short = {
        "inactive": "I",
        "traditional_only": "T",
        "emerging_only": "E",
        "both": "B",
    }
    return f"{short[from_state]}->{short[to_state]}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_months = set(args.exclude_month or [])

    eligible_users = load_eligible_users(Path(args.commenters_json), args.include_single_commenters)
    video_media_types = load_video_media_types(Path(args.videos_json))

    month_user_media = defaultdict(set)
    month_user_comments = Counter()
    monthly_media_users = defaultdict(lambda: defaultdict(set))
    monthly_media_comments = defaultdict(Counter)
    skipped = Counter()
    total_records = 0

    with Path(args.comments_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            row = json.loads(line)
            month = row.get("month") or ""
            if month and month > args.max_month:
                skipped["after_max_month"] += 1
                continue
            if month in excluded_months:
                skipped["excluded_month"] += 1
                continue
            user_id = row.get("user_id")
            if not user_id or user_id not in eligible_users:
                skipped["single_comment_or_missing_user"] += 1
                continue
            media_type = video_media_types.get(row.get("video_id"), "unknown")
            if media_type not in MEDIA_TYPE_ORDER:
                skipped["unknown_media_type"] += 1
                continue

            key = (month, user_id)
            month_user_media[key].add(media_type)
            month_user_comments[key] += 1
            monthly_media_users[month][media_type].add(user_id)
            monthly_media_comments[month][media_type] += 1

    months = sorted({month for month, _ in month_user_media})
    month_user_state = defaultdict(dict)
    for (month, user_id), media_types in month_user_media.items():
        month_user_state[month][user_id] = state_from_media_set(media_types)

    state_rows = []
    overlap_rows = []
    for month in months:
        state_counts = Counter(month_user_state[month].values())
        state_comment_counts = Counter()
        active_users = set(month_user_state[month])
        active_comment_count = sum(month_user_comments[(month, user_id)] for user_id in active_users)
        for user_id, state in month_user_state[month].items():
            state_comment_counts[state] += month_user_comments[(month, user_id)]

        for state in STATE_ORDER:
            user_count = state_counts[state]
            comment_count = state_comment_counts[state]
            state_rows.append(
                {
                    "month": month,
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "user_count": user_count,
                    "user_share": f"{safe_ratio(user_count, len(active_users)):.6f}",
                    "comment_count": comment_count,
                    "comment_share": f"{safe_ratio(comment_count, active_comment_count):.6f}",
                    "active_user_count": len(active_users),
                    "active_comment_count": active_comment_count,
                }
            )

        traditional_users = monthly_media_users[month]["traditional"]
        emerging_users = monthly_media_users[month]["emerging"]
        both_users = traditional_users & emerging_users
        union_users = traditional_users | emerging_users
        overlap_rows.append(
            {
                "month": month,
                "traditional_user_count": len(traditional_users),
                "emerging_user_count": len(emerging_users),
                "overlap_user_count": len(both_users),
                "union_user_count": len(union_users),
                "jaccard_overlap": f"{safe_ratio(len(both_users), len(union_users)):.6f}",
                "overlap_share_of_traditional": f"{safe_ratio(len(both_users), len(traditional_users)):.6f}",
                "overlap_share_of_emerging": f"{safe_ratio(len(both_users), len(emerging_users)):.6f}",
                "traditional_comment_count": monthly_media_comments[month]["traditional"],
                "emerging_comment_count": monthly_media_comments[month]["emerging"],
            }
        )

    transition_rows = []
    churn_rows = []
    mobility_rows = []
    overall_active_transition_counts = Counter()
    overall_churn_transition_counts = Counter()

    for from_month, to_month in zip(months, months[1:]):
        from_users = set(month_user_state[from_month])
        to_users = set(month_user_state[to_month])
        continuing_users = from_users & to_users
        union_users = from_users | to_users

        active_counts = Counter()
        from_state_counts = Counter()
        for user_id in continuing_users:
            from_state = month_user_state[from_month][user_id]
            to_state = month_user_state[to_month][user_id]
            active_counts[(from_state, to_state)] += 1
            from_state_counts[from_state] += 1
            overall_active_transition_counts[(from_state, to_state)] += 1

        churn_counts = Counter()
        churn_from_state_counts = Counter()
        for user_id in union_users:
            from_state = month_user_state[from_month].get(user_id, "inactive")
            to_state = month_user_state[to_month].get(user_id, "inactive")
            churn_counts[(from_state, to_state)] += 1
            churn_from_state_counts[from_state] += 1
            overall_churn_transition_counts[(from_state, to_state)] += 1

        for from_state in STATE_ORDER:
            for to_state in STATE_ORDER:
                count = active_counts[(from_state, to_state)]
                transition_rows.append(
                    {
                        "from_month": from_month,
                        "to_month": to_month,
                        "month_pair": f"{from_month}->{to_month}",
                        "from_state": from_state,
                        "from_state_label": STATE_LABELS[from_state],
                        "to_state": to_state,
                        "to_state_label": STATE_LABELS[to_state],
                        "transition": transition_label(from_state, to_state),
                        "user_count": count,
                        "row_share": f"{safe_ratio(count, from_state_counts[from_state]):.6f}",
                        "pair_share": f"{safe_ratio(count, len(continuing_users)):.6f}",
                        "continuing_user_count": len(continuing_users),
                    }
                )

        for from_state in CHURN_STATE_ORDER:
            for to_state in CHURN_STATE_ORDER:
                count = churn_counts[(from_state, to_state)]
                churn_rows.append(
                    {
                        "from_month": from_month,
                        "to_month": to_month,
                        "month_pair": f"{from_month}->{to_month}",
                        "from_state": from_state,
                        "from_state_label": STATE_LABELS[from_state],
                        "to_state": to_state,
                        "to_state_label": STATE_LABELS[to_state],
                        "transition": transition_label(from_state, to_state),
                        "user_count": count,
                        "row_share": f"{safe_ratio(count, churn_from_state_counts[from_state]):.6f}",
                        "pair_share": f"{safe_ratio(count, len(union_users)):.6f}",
                        "union_user_count": len(union_users),
                    }
                )

        same_state_count = sum(active_counts[(state, state)] for state in STATE_ORDER)
        side_switch_count = (
            active_counts[("traditional_only", "emerging_only")]
            + active_counts[("emerging_only", "traditional_only")]
        )
        single_to_both_count = active_counts[("traditional_only", "both")] + active_counts[("emerging_only", "both")]
        both_to_single_count = active_counts[("both", "traditional_only")] + active_counts[("both", "emerging_only")]
        both_to_both_count = active_counts[("both", "both")]
        active_to_inactive_count = sum(churn_counts[(state, "inactive")] for state in STATE_ORDER)
        inactive_to_active_count = sum(churn_counts[("inactive", state)] for state in STATE_ORDER)
        retained_count = len(continuing_users)

        mobility_rows.append(
            {
                "from_month": from_month,
                "to_month": to_month,
                "month_pair": f"{from_month}->{to_month}",
                "continuing_user_count": len(continuing_users),
                "union_user_count": len(union_users),
                "same_state_count": same_state_count,
                "same_state_share": f"{safe_ratio(same_state_count, len(continuing_users)):.6f}",
                "state_change_count": len(continuing_users) - same_state_count,
                "state_change_share": f"{safe_ratio(len(continuing_users) - same_state_count, len(continuing_users)):.6f}",
                "side_switch_count": side_switch_count,
                "side_switch_share": f"{safe_ratio(side_switch_count, len(continuing_users)):.6f}",
                "single_to_both_count": single_to_both_count,
                "single_to_both_share": f"{safe_ratio(single_to_both_count, len(continuing_users)):.6f}",
                "both_to_single_count": both_to_single_count,
                "both_to_single_share": f"{safe_ratio(both_to_single_count, len(continuing_users)):.6f}",
                "both_to_both_count": both_to_both_count,
                "both_to_both_share": f"{safe_ratio(both_to_both_count, len(continuing_users)):.6f}",
                "active_to_inactive_count": active_to_inactive_count,
                "active_to_inactive_share_of_union": f"{safe_ratio(active_to_inactive_count, len(union_users)):.6f}",
                "inactive_to_active_count": inactive_to_active_count,
                "inactive_to_active_share_of_union": f"{safe_ratio(inactive_to_active_count, len(union_users)):.6f}",
                "retained_user_count": retained_count,
                "retained_share_of_union": f"{safe_ratio(retained_count, len(union_users)):.6f}",
            }
        )

    overall_transition_rows = []
    for from_state in STATE_ORDER:
        row_total = sum(overall_active_transition_counts[(from_state, to_state)] for to_state in STATE_ORDER)
        for to_state in STATE_ORDER:
            count = overall_active_transition_counts[(from_state, to_state)]
            overall_transition_rows.append(
                {
                    "matrix_type": "continuing_active_users",
                    "from_state": from_state,
                    "from_state_label": STATE_LABELS[from_state],
                    "to_state": to_state,
                    "to_state_label": STATE_LABELS[to_state],
                    "transition": transition_label(from_state, to_state),
                    "user_count": count,
                    "row_share": f"{safe_ratio(count, row_total):.6f}",
                }
            )

    for from_state in CHURN_STATE_ORDER:
        row_total = sum(overall_churn_transition_counts[(from_state, to_state)] for to_state in CHURN_STATE_ORDER)
        for to_state in CHURN_STATE_ORDER:
            count = overall_churn_transition_counts[(from_state, to_state)]
            overall_transition_rows.append(
                {
                    "matrix_type": "including_inactive_state",
                    "from_state": from_state,
                    "from_state_label": STATE_LABELS[from_state],
                    "to_state": to_state,
                    "to_state_label": STATE_LABELS[to_state],
                    "transition": transition_label(from_state, to_state),
                    "user_count": count,
                    "row_share": f"{safe_ratio(count, row_total):.6f}",
                }
            )

    state_fields = [
        "month", "state", "state_label", "user_count", "user_share",
        "comment_count", "comment_share", "active_user_count", "active_comment_count",
    ]
    overlap_fields = [
        "month", "traditional_user_count", "emerging_user_count", "overlap_user_count",
        "union_user_count", "jaccard_overlap", "overlap_share_of_traditional",
        "overlap_share_of_emerging", "traditional_comment_count", "emerging_comment_count",
    ]
    transition_fields = [
        "from_month", "to_month", "month_pair", "from_state", "from_state_label",
        "to_state", "to_state_label", "transition", "user_count", "row_share",
        "pair_share", "continuing_user_count",
    ]
    churn_fields = [
        "from_month", "to_month", "month_pair", "from_state", "from_state_label",
        "to_state", "to_state_label", "transition", "user_count", "row_share",
        "pair_share", "union_user_count",
    ]
    mobility_fields = [
        "from_month", "to_month", "month_pair", "continuing_user_count", "union_user_count",
        "same_state_count", "same_state_share", "state_change_count", "state_change_share",
        "side_switch_count", "side_switch_share", "single_to_both_count", "single_to_both_share",
        "both_to_single_count", "both_to_single_share", "both_to_both_count", "both_to_both_share",
        "active_to_inactive_count", "active_to_inactive_share_of_union",
        "inactive_to_active_count", "inactive_to_active_share_of_union",
        "retained_user_count", "retained_share_of_union",
    ]
    overall_fields = [
        "matrix_type", "from_state", "from_state_label", "to_state", "to_state_label",
        "transition", "user_count", "row_share",
    ]

    write_csv(output_dir / "monthly_user_media_state_summary.csv", state_rows, state_fields)
    write_csv(output_dir / "monthly_user_overlap_summary.csv", overlap_rows, overlap_fields)
    write_csv(output_dir / "monthly_user_transition_summary.csv", transition_rows, transition_fields)
    write_csv(output_dir / "monthly_user_transition_churn_summary.csv", churn_rows, churn_fields)
    write_csv(output_dir / "monthly_user_mobility_index.csv", mobility_rows, mobility_fields)
    write_csv(output_dir / "monthly_user_transition_overall_matrix.csv", overall_transition_rows, overall_fields)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": Path(__file__).name,
        "comments_jsonl": str(args.comments_jsonl),
        "commenters_json": str(args.commenters_json),
        "videos_json": str(args.videos_json),
        "max_month": args.max_month,
        "excluded_months": sorted(excluded_months),
        "include_single_commenters": args.include_single_commenters,
        "total_records": total_records,
        "included_user_months": len(month_user_media),
        "months": months,
        "state_order": STATE_ORDER,
        "churn_state_order": CHURN_STATE_ORDER,
        "state_labels": STATE_LABELS,
        "media_type_labels": MEDIA_TYPE_LABELS,
        "skipped": dict(skipped),
        "outputs": {
            "monthly_state_csv": str(output_dir / "monthly_user_media_state_summary.csv"),
            "monthly_overlap_csv": str(output_dir / "monthly_user_overlap_summary.csv"),
            "monthly_transition_csv": str(output_dir / "monthly_user_transition_summary.csv"),
            "monthly_transition_churn_csv": str(output_dir / "monthly_user_transition_churn_summary.csv"),
            "monthly_mobility_index_csv": str(output_dir / "monthly_user_mobility_index.csv"),
            "overall_transition_matrix_csv": str(output_dir / "monthly_user_transition_overall_matrix.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'monthly_user_media_state_summary.csv'}")
    print(f"Wrote {output_dir / 'monthly_user_overlap_summary.csv'}")
    print(f"Wrote {output_dir / 'monthly_user_transition_summary.csv'}")
    print(f"Wrote {output_dir / 'monthly_user_transition_churn_summary.csv'}")
    print(f"Wrote {output_dir / 'monthly_user_mobility_index.csv'}")
    print(f"Wrote {output_dir / 'monthly_user_transition_overall_matrix.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
