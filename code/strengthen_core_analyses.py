#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strengthen the observational evidence behind the topic-emotion-mobility paper.

This script leaves the existing descriptive analyses untouched and adds four
complementary layers:
  1. Time-varying commenter tiers based only on activity observed up to month t.
  2. Topic/emotion trend interactions and user-month discourse-to-mobility models.
  3. Channel leave-one-out and equal-channel-weight sensitivity checks.
  4. Stratified templates for manual corpus-relevance and multi-label topic validation.

All outputs are observational.  They describe associations in observed YouTube
commenting activity and are not causal estimates of media effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from analyze_emotion_by_media_type import EMOTION_LABELS_EN, EMOTION_ORDER
from analyze_topic_by_media_type import TOPICS, normalize_text, topic_hits
from channel_media_types import (
    MEDIA_TYPE_LABELS,
    display_label_for_channel,
    media_type_for_channel,
)
from plot_topic_intensity_per_1000_views_by_media_type import TOPIC_LABELS_EN


BASE_DIR = Path(__file__).resolve().parent
COMMENTS_JSONL = BASE_DIR / "user_comment_emotions.jsonl"
VIDEOS_JSON = BASE_DIR / "all_filtered_videos_by_keywords.json"
OUTPUT_DIR = BASE_DIR / "analysis_strengthening"
PAPER_SUPPLEMENTARY_DIR = BASE_DIR / "paper_selected_figures" / "supplementary" / "analysis_strengthening"

MEDIA_TYPES = ("traditional", "emerging")
TIER_ORDER = ("low_2_5", "mid_6_20", "core_21_50", "high_51_plus")
TIER_LABELS = {
    "low_2_5": "2-5 comments",
    "mid_6_20": "6-20 comments",
    "core_21_50": "21-50 comments",
    "high_51_plus": "51+ comments",
}
STATE_ORDER = ("traditional_only", "emerging_only", "both", "inactive")
STATE_LABELS = {
    "traditional_only": "Traditional only",
    "emerging_only": "Emerging only",
    "both": "Both media types",
    "inactive": "Inactive",
}
TOPIC_ORDER = tuple(TOPICS)
TOPIC_INDEX = {topic: index for index, topic in enumerate(TOPIC_ORDER)}
EMOTION_INDEX = {emotion: index for index, emotion in enumerate(EMOTION_ORDER)}
TOPIC_LABELS_CLEAN = {topic: TOPIC_LABELS_EN[topic].replace("\n", " ") for topic in TOPIC_ORDER}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add dynamic-tier, linkage, robustness, and validation analyses."
    )
    parser.add_argument("--comments-jsonl", type=Path, default=COMMENTS_JSONL)
    parser.add_argument("--videos-json", type=Path, default=VIDEOS_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--paper-supplementary-dir", type=Path, default=PAPER_SUPPLEMENTARY_DIR)
    parser.add_argument("--max-month", default="2025-10")
    parser.add_argument("--exclude-month", action="append", default=["2025-11"])
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--corpus-sample-per-stratum", type=int, default=15)
    parser.add_argument("--topic-sample-per-stratum", type=int, default=20)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft JhengHei",
        "PingFang TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_json_payload(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def parse_count(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def month_from_timestamp(value: str) -> str:
    return (value or "")[:7]


def quarter_from_month(month: str) -> str:
    if len(month) != 7:
        return "Unknown"
    year, month_number = month.split("-")
    return f"{year}-Q{(int(month_number) - 1) // 3 + 1}"


def assign_tier(cumulative_comment_count: int) -> str | None:
    if cumulative_comment_count <= 1:
        return None
    if cumulative_comment_count <= 5:
        return "low_2_5"
    if cumulative_comment_count <= 20:
        return "mid_6_20"
    if cumulative_comment_count <= 50:
        return "core_21_50"
    return "high_51_plus"


def state_from_media_types(media_types: set[str]) -> str:
    if media_types == {"traditional"}:
        return "traditional_only"
    if media_types == {"emerging"}:
        return "emerging_only"
    if media_types == {"traditional", "emerging"}:
        return "both"
    return "inactive"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normal_p_value(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank in range(len(order), 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def solve_weighted_lstsq(x_matrix: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root_weights = np.sqrt(np.maximum(weights, 1e-12))
    return np.linalg.lstsq(x_matrix * root_weights[:, None], outcome * root_weights, rcond=None)[0]


def fit_grouped_binomial(
    x_matrix: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit a grouped logistic model with quasi-binomial standard errors."""
    beta = np.zeros(x_matrix.shape[1], dtype=float)
    observed_rate = successes / np.maximum(trials, 1.0)
    for _ in range(200):
        eta = x_matrix @ beta
        probability = np.clip(sigmoid(eta), 1e-8, 1 - 1e-8)
        weights = trials * probability * (1 - probability)
        working_response = eta + (observed_rate - probability) / np.maximum(probability * (1 - probability), 1e-12)
        beta_new = solve_weighted_lstsq(x_matrix, working_response, weights)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    probability = np.clip(sigmoid(x_matrix @ beta), 1e-8, 1 - 1e-8)
    weights = trials * probability * (1 - probability)
    bread = np.linalg.pinv(x_matrix.T @ (weights[:, None] * x_matrix))
    pearson = float(np.sum((successes - trials * probability) ** 2 / np.maximum(weights, 1e-12)))
    dispersion = pearson / max(1, x_matrix.shape[0] - x_matrix.shape[1])
    return {
        "beta": beta,
        "covariance": bread * dispersion,
        "feature_names": feature_names,
        "n_obs": int(x_matrix.shape[0]),
        "df_resid": int(max(1, x_matrix.shape[0] - x_matrix.shape[1])),
        "dispersion": dispersion,
    }


def fit_clustered_logistic(
    x_matrix: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit a Bernoulli logit model with a sandwich covariance clustered by user."""
    beta = np.zeros(x_matrix.shape[1], dtype=float)
    for _ in range(200):
        eta = x_matrix @ beta
        probability = np.clip(sigmoid(eta), 1e-8, 1 - 1e-8)
        weights = probability * (1 - probability)
        working_response = eta + (outcome - probability) / np.maximum(weights, 1e-12)
        beta_new = solve_weighted_lstsq(x_matrix, working_response, weights)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    probability = np.clip(sigmoid(x_matrix @ beta), 1e-8, 1 - 1e-8)
    weights = probability * (1 - probability)
    bread = np.linalg.pinv(x_matrix.T @ (weights[:, None] * x_matrix))
    scores = (outcome - probability)[:, None] * x_matrix
    unique_clusters, cluster_indices = np.unique(clusters, return_inverse=True)
    grouped_scores = np.zeros((len(unique_clusters), x_matrix.shape[1]), dtype=float)
    np.add.at(grouped_scores, cluster_indices, scores)
    meat = grouped_scores.T @ grouped_scores
    covariance = bread @ meat @ bread
    return {
        "beta": beta,
        "covariance": covariance,
        "feature_names": feature_names,
        "n_obs": int(x_matrix.shape[0]),
        "cluster_count": int(len(unique_clusters)),
    }


def term_rows(model_name: str, fit: dict, scale_name: str = "odds_ratio") -> list[dict]:
    rows = []
    for index, term in enumerate(fit["feature_names"]):
        estimate = float(fit["beta"][index])
        standard_error = float(math.sqrt(max(fit["covariance"][index, index], 0.0)))
        z_value = estimate / standard_error if standard_error else 0.0
        p_value = normal_p_value(z_value)
        rows.append(
            {
                "model": model_name,
                "term": term,
                "estimate_log_odds": estimate,
                "std_error": standard_error,
                "z_value": z_value,
                "p_value": p_value,
                scale_name: math.exp(estimate),
                f"{scale_name}_ci_lower": math.exp(estimate - 1.96 * standard_error),
                f"{scale_name}_ci_upper": math.exp(estimate + 1.96 * standard_error),
                "n_obs": fit["n_obs"],
            }
        )
    adjusted = bh_adjust([row["p_value"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_value_bh"] = p_adjusted
    return rows


def reservoir_add(reservoir: list[dict], seen: int, candidate: dict, capacity: int, rng: random.Random) -> None:
    if len(reservoir) < capacity:
        reservoir.append(candidate)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < capacity:
        reservoir[replacement_index] = candidate


def load_videos(path: Path) -> dict[str, dict]:
    videos = {}
    for video in read_json_payload(path):
        video_id = video.get("id")
        snippet = video.get("snippet") or {}
        statistics = video.get("statistics") or {}
        channel = snippet.get("channelTitle") or video.get("channelTitle") or ""
        media_type = media_type_for_channel(channel)
        if not video_id or media_type not in MEDIA_TYPES:
            continue
        videos[video_id] = {
            "video_id": video_id,
            "channel": channel,
            "channel_label": display_label_for_channel(channel),
            "media_type": media_type,
            "published_at": snippet.get("publishedAt") or "",
            "published_month": month_from_timestamp(snippet.get("publishedAt") or ""),
            "title": snippet.get("title") or "",
            "description": snippet.get("description") or "",
            "matched_keywords": "; ".join(video.get("matched_keywords") or []),
            "view_count": parse_count(statistics.get("viewCount")),
            "platform_comment_count": parse_count(statistics.get("commentCount")),
        }
    return videos


def new_user_month_record() -> dict:
    return {
        "comment_count": 0,
        "media_types": set(),
        "topic_counts": np.zeros(len(TOPIC_ORDER), dtype=np.int32),
        "emotion_counts": np.zeros(len(EMOTION_ORDER), dtype=np.int32),
        "by_media": {
            media_type: {
                "comment_count": 0,
                "topic_counts": np.zeros(len(TOPIC_ORDER), dtype=np.int32),
                "emotion_counts": np.zeros(len(EMOTION_ORDER), dtype=np.int32),
            }
            for media_type in MEDIA_TYPES
        },
    }


def primary_topic(topic_counts: np.ndarray) -> str:
    if not int(topic_counts.sum()):
        return "No identified topic"
    return TOPIC_ORDER[int(np.argmax(topic_counts))]


def primary_emotion(emotion_counts: np.ndarray) -> str:
    if not int(emotion_counts.sum()):
        return "Neutral"
    return EMOTION_ORDER[int(np.argmax(emotion_counts))]


def load_comment_data(
    comments_path: Path,
    videos: dict[str, dict],
    max_month: str,
    excluded_months: set[str],
    topic_sample_per_stratum: int,
    rng: random.Random,
) -> tuple[dict[str, dict[str, dict]], dict, dict, dict, dict, Counter]:
    """Read comments once and assemble the structures needed by every analysis."""
    month_users: dict[str, dict[str, dict]] = defaultdict(dict)
    channel_topic_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(TOPIC_ORDER), dtype=np.int64))
    channel_emotion_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(EMOTION_ORDER), dtype=np.int64))
    channel_comment_counts: Counter = Counter()
    channel_users: dict[str, set[str]] = defaultdict(set)
    video_observed_comments: Counter = Counter()
    topic_reservoirs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    topic_reservoir_seen: Counter = Counter()
    skipped: Counter = Counter()

    with comments_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            month = row.get("month") or ""
            if not month or month > max_month:
                skipped["after_max_month_or_missing_month"] += 1
                continue
            if month in excluded_months:
                skipped["excluded_month"] += 1
                continue
            video = videos.get(row.get("video_id"))
            if not video:
                skipped["unknown_or_out_of_scope_video"] += 1
                continue
            user_id = row.get("user_id") or ""
            if not user_id:
                skipped["missing_user_id"] += 1
                continue

            media_type = video["media_type"]
            channel = video["channel"]
            text = normalize_text(row.get("text") or "")
            hits, _ = topic_hits(text)
            emotion = EMOTION_LABELS_EN.get(row.get("emotion_label"), "Neutral")

            user_record = month_users[month].setdefault(user_id, new_user_month_record())
            user_record["comment_count"] += 1
            user_record["media_types"].add(media_type)
            user_record["by_media"][media_type]["comment_count"] += 1
            for topic in hits:
                topic_index = TOPIC_INDEX[topic]
                user_record["topic_counts"][topic_index] += 1
                user_record["by_media"][media_type]["topic_counts"][topic_index] += 1
                channel_topic_counts[channel][topic_index] += 1
            emotion_index = EMOTION_INDEX[emotion]
            user_record["emotion_counts"][emotion_index] += 1
            user_record["by_media"][media_type]["emotion_counts"][emotion_index] += 1
            channel_emotion_counts[channel][emotion_index] += 1
            channel_comment_counts[channel] += 1
            channel_users[channel].add(user_id)
            video_observed_comments[row.get("video_id")] += 1

            comment_topic_counts = np.zeros(len(TOPIC_ORDER), dtype=np.int16)
            for topic in hits:
                comment_topic_counts[TOPIC_INDEX[topic]] = 1
            auto_primary_topic = primary_topic(comment_topic_counts)
            topic_stratum = (media_type, auto_primary_topic)
            topic_reservoir_seen[topic_stratum] += 1
            candidate = {
                "comment_id": row.get("comment_id") or "",
                "video_id": row.get("video_id") or "",
                "month": month,
                "media_type": media_type,
                "channel": channel,
                "channel_label": video["channel_label"],
                "comment_text": text,
                "auto_topic_labels": "; ".join(topic for topic in TOPIC_ORDER if topic in hits),
                "auto_primary_topic": auto_primary_topic,
                "auto_emotion": emotion,
            }
            reservoir_add(
                topic_reservoirs[topic_stratum],
                topic_reservoir_seen[topic_stratum],
                candidate,
                topic_sample_per_stratum,
                rng,
            )

    return (
        month_users,
        {
            "topic_counts": channel_topic_counts,
            "emotion_counts": channel_emotion_counts,
            "comment_counts": channel_comment_counts,
            "users": channel_users,
            "video_observed_comments": video_observed_comments,
        },
        topic_reservoirs,
        topic_reservoir_seen,
        dict(skipped),
        skipped,
    )


def finalize_dynamic_tiers(month_users: dict[str, dict[str, dict]]) -> tuple[dict[str, dict[str, dict]], list[dict], dict]:
    cumulative_comments: Counter = Counter()
    state_rows: list[dict] = []
    temporal_counts: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "comment_count": 0,
            "topic_counts": np.zeros(len(TOPIC_ORDER), dtype=np.int64),
            "emotion_counts": np.zeros(len(EMOTION_ORDER), dtype=np.int64),
        }
    )
    dynamic_summary = Counter()

    for month in sorted(month_users):
        for user_id, record in month_users[month].items():
            cumulative_comments[user_id] += record["comment_count"]
            record["dynamic_tier"] = assign_tier(cumulative_comments[user_id])
            record["state"] = state_from_media_types(record["media_types"])
            record["cumulative_comment_count"] = cumulative_comments[user_id]
            record["dominant_topic"] = primary_topic(record["topic_counts"])
            record["dominant_emotion"] = primary_emotion(record["emotion_counts"])
            record["month_comment_count"] = record["comment_count"]

            tier = record["dynamic_tier"]
            if not tier:
                dynamic_summary["user_months_below_two_comments"] += 1
                continue
            dynamic_summary["eligible_user_months"] += 1
            dynamic_summary[f"tier::{tier}"] += 1
            state_rows.append(
                {
                    "month": month,
                    "user_id": user_id,
                    "dynamic_tier": tier,
                    "dynamic_tier_label": TIER_LABELS[tier],
                    "cumulative_comment_count": cumulative_comments[user_id],
                    "month_comment_count": record["comment_count"],
                    "state": record["state"],
                    "state_label": STATE_LABELS[record["state"]],
                    "dominant_topic": record["dominant_topic"],
                    "dominant_emotion": record["dominant_emotion"],
                }
            )
            for media_type in MEDIA_TYPES:
                by_media = record["by_media"][media_type]
                if not by_media["comment_count"]:
                    continue
                cell = temporal_counts[(month, media_type, tier)]
                cell["comment_count"] += by_media["comment_count"]
                cell["topic_counts"] += by_media["topic_counts"]
                cell["emotion_counts"] += by_media["emotion_counts"]

    return month_users, state_rows, {"temporal_counts": temporal_counts, "summary": dict(dynamic_summary)}


def build_dynamic_transition_rows(month_users: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    months = sorted(month_users)
    transition_rows = []
    model_rows = []
    for from_month, to_month in zip(months, months[1:]):
        grouped: Counter = Counter()
        for user_id, current in month_users[from_month].items():
            tier = current.get("dynamic_tier")
            if not tier:
                continue
            from_state = current["state"]
            next_record = month_users[to_month].get(user_id)
            to_state = next_record["state"] if next_record else "inactive"
            grouped[(tier, from_state, to_state)] += 1

        for (tier, from_state, to_state), count in sorted(grouped.items()):
            total = sum(
                grouped[(tier, from_state, destination)]
                for destination in STATE_ORDER
                if (tier, from_state, destination) in grouped
            )
            transition_rows.append(
                {
                    "from_month": from_month,
                    "to_month": to_month,
                    "month_pair": f"{from_month}->{to_month}",
                    "dynamic_tier": tier,
                    "dynamic_tier_label": TIER_LABELS[tier],
                    "from_state": from_state,
                    "from_state_label": STATE_LABELS[from_state],
                    "to_state": to_state,
                    "to_state_label": STATE_LABELS[to_state],
                    "transition_count": count,
                    "from_state_total": total,
                    "transition_rate": safe_ratio(count, total),
                }
            )

        for tier in TIER_ORDER:
            for from_state in STATE_ORDER[:-1]:
                total = sum(grouped[(tier, from_state, destination)] for destination in STATE_ORDER)
                if not total:
                    continue
                common = {
                    "from_month": from_month,
                    "to_month": to_month,
                    "month_pair": f"{from_month}->{to_month}",
                    "dynamic_tier": tier,
                    "from_state": from_state,
                    "trials": total,
                }
                model_rows.append({**common, "outcome": "next_month_inactivity", "successes": grouped[(tier, from_state, "inactive")]})
                model_rows.append({**common, "outcome": "same_state_retention", "successes": grouped[(tier, from_state, from_state)]})
                if from_state in {"traditional_only", "emerging_only"}:
                    other_state = "emerging_only" if from_state == "traditional_only" else "traditional_only"
                    model_rows.append({**common, "outcome": "single_to_both", "successes": grouped[(tier, from_state, "both")]})
                    model_rows.append({**common, "outcome": "direct_media_switch", "successes": grouped[(tier, from_state, other_state)]})
    return transition_rows, model_rows


def transition_design_rows(rows: list[dict], outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in rows if row["outcome"] == outcome]
    state_levels = ["traditional_only", "emerging_only"] if outcome in {"single_to_both", "direct_media_switch"} else ["traditional_only", "emerging_only", "both"]
    pairs = sorted({row["month_pair"] for row in selected})
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    centered_time = (len(pairs) - 1) / 2
    feature_names = ["Intercept"]
    feature_names += ["Source: Emerging-only vs Traditional-only"]
    if len(state_levels) == 3:
        feature_names += ["Source: Both vs Traditional-only"]
    feature_names += [f"Dynamic tier: {TIER_LABELS[tier]} vs {TIER_LABELS[TIER_ORDER[0]]}" for tier in TIER_ORDER[1:]]
    feature_names += ["Monthly time index"]
    x_values = []
    successes = []
    trials = []
    for row in selected:
        values = [1.0, float(row["from_state"] == "emerging_only")]
        if len(state_levels) == 3:
            values.append(float(row["from_state"] == "both"))
        values += [float(row["dynamic_tier"] == tier) for tier in TIER_ORDER[1:]]
        values.append(float(pair_index[row["month_pair"]] - centered_time))
        x_values.append(values)
        successes.append(float(row["successes"]))
        trials.append(float(row["trials"]))
    return np.asarray(x_values), np.asarray(successes), np.asarray(trials), feature_names


def fit_dynamic_transition_models(model_rows: list[dict]) -> list[dict]:
    output_rows = []
    for outcome in ("single_to_both", "direct_media_switch", "same_state_retention", "next_month_inactivity"):
        x_matrix, successes, trials, feature_names = transition_design_rows(model_rows, outcome)
        if not len(x_matrix):
            continue
        fit = fit_grouped_binomial(x_matrix, successes, trials, feature_names)
        rows = term_rows(f"dynamic_tier_{outcome}", fit)
        for row in rows:
            row["outcome"] = outcome
            row["dispersion"] = fit["dispersion"]
            row["total_trials"] = int(trials.sum())
            row["total_successes"] = int(successes.sum())
        output_rows.extend(rows)
    return output_rows


def build_user_month_linkage_records(month_users: dict[str, dict[str, dict]]) -> list[dict]:
    records = []
    months = sorted(month_users)
    user_codes: dict[str, int] = {}
    for month_index, (from_month, to_month) in enumerate(zip(months, months[1:])):
        for user_id, current in month_users[from_month].items():
            tier = current.get("dynamic_tier")
            if not tier:
                continue
            if user_id not in user_codes:
                user_codes[user_id] = len(user_codes)
            next_record = month_users[to_month].get(user_id)
            next_state = next_record["state"] if next_record else "inactive"
            records.append(
                {
                    "user_code": user_codes[user_id],
                    "month_index": month_index,
                    "from_state": current["state"],
                    "next_state": next_state,
                    "dynamic_tier": tier,
                    "dominant_topic": current["dominant_topic"],
                    "dominant_emotion": current["dominant_emotion"],
                    "log_month_comment_count": math.log1p(current["month_comment_count"]),
                }
            )
    return records


def linkage_design(
    records: list[dict],
    outcome: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if outcome == "next_month_both_from_single":
        selected = [record for record in records if record["from_state"] in {"traditional_only", "emerging_only"}]
        outcome_value = lambda record: float(record["next_state"] == "both")
        source_names = ["Source: Emerging-only vs Traditional-only"]
    elif outcome == "next_month_inactivity":
        selected = records
        outcome_value = lambda record: float(record["next_state"] == "inactive")
        source_names = ["Source: Emerging-only vs Traditional-only", "Source: Both vs Traditional-only"]
    else:
        raise ValueError(f"Unknown linkage outcome: {outcome}")

    mean_time = np.mean([record["month_index"] for record in selected]) if selected else 0.0
    feature_names = ["Intercept", *source_names]
    feature_names += [f"Dynamic tier: {TIER_LABELS[tier]} vs {TIER_LABELS[TIER_ORDER[0]]}" for tier in TIER_ORDER[1:]]
    feature_names.append("log(1 + comments in current month)")
    feature_names += [f"Topic: {TOPIC_LABELS_CLEAN[topic]} vs No identified topic" for topic in TOPIC_ORDER]
    feature_names += [f"Emotion: {emotion} vs Neutral" for emotion in EMOTION_ORDER if emotion != "Neutral"]
    feature_names.append("Monthly time index")

    x_values = []
    outcomes = []
    clusters = []
    for record in selected:
        values = [1.0, float(record["from_state"] == "emerging_only")]
        if outcome == "next_month_inactivity":
            values.append(float(record["from_state"] == "both"))
        values += [float(record["dynamic_tier"] == tier) for tier in TIER_ORDER[1:]]
        values.append(record["log_month_comment_count"])
        values += [float(record["dominant_topic"] == topic) for topic in TOPIC_ORDER]
        values += [float(record["dominant_emotion"] == emotion) for emotion in EMOTION_ORDER if emotion != "Neutral"]
        values.append(record["month_index"] - mean_time)
        x_values.append(values)
        outcomes.append(outcome_value(record))
        clusters.append(record["user_code"])
    return np.asarray(x_values), np.asarray(outcomes), np.asarray(clusters), feature_names


def fit_linkage_models(records: list[dict]) -> list[dict]:
    rows = []
    for outcome in ("next_month_both_from_single", "next_month_inactivity"):
        x_matrix, values, clusters, feature_names = linkage_design(records, outcome)
        fit = fit_clustered_logistic(x_matrix, values, clusters, feature_names)
        model_rows = term_rows(f"user_month_{outcome}", fit)
        for row in model_rows:
            row["outcome"] = outcome
            row["cluster_count"] = fit["cluster_count"]
            row["outcome_rate"] = float(values.mean())
        rows.extend(model_rows)
    return rows


def build_temporal_cells(temporal_counts: dict) -> tuple[list[dict], list[dict]]:
    topic_rows = []
    emotion_rows = []
    months = sorted({key[0] for key in temporal_counts})
    month_index = {month: index for index, month in enumerate(months)}
    for (month, media_type, tier), counts in sorted(temporal_counts.items()):
        total = counts["comment_count"]
        if not total:
            continue
        common = {
            "month": month,
            "month_index": month_index[month],
            "media_type": media_type,
            "dynamic_tier": tier,
            "comment_count": total,
        }
        for index, topic in enumerate(TOPIC_ORDER):
            topic_rows.append({**common, "category": topic, "successes": int(counts["topic_counts"][index]), "family": "topic"})
        for index, emotion in enumerate(EMOTION_ORDER):
            emotion_rows.append({**common, "category": emotion, "successes": int(counts["emotion_counts"][index]), "family": "emotion"})
    return topic_rows, emotion_rows


def fit_trend_interactions(rows: list[dict], family: str) -> list[dict]:
    output_rows = []
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        center = np.mean([row["month_index"] for row in selected])
        x_values = []
        successes = []
        trials = []
        for row in selected:
            emerging = float(row["media_type"] == "emerging")
            time = row["month_index"] - center
            x_values.append(
                [
                    1.0,
                    emerging,
                    *[float(row["dynamic_tier"] == tier) for tier in TIER_ORDER[1:]],
                    time,
                    emerging * time,
                ]
            )
            successes.append(row["successes"])
            trials.append(row["comment_count"])
        feature_names = [
            "Intercept",
            "Emerging vs Traditional",
            *[f"Dynamic tier: {TIER_LABELS[tier]} vs {TIER_LABELS[TIER_ORDER[0]]}" for tier in TIER_ORDER[1:]],
            "Monthly trend in Traditional",
            "Emerging x monthly trend",
        ]
        fit = fit_grouped_binomial(np.asarray(x_values), np.asarray(successes), np.asarray(trials), feature_names)
        interaction_index = feature_names.index("Emerging x monthly trend")
        estimate = float(fit["beta"][interaction_index])
        standard_error = float(math.sqrt(max(fit["covariance"][interaction_index, interaction_index], 0.0)))
        z_value = estimate / standard_error if standard_error else 0.0
        output_rows.append(
            {
                "family": family,
                "category": category,
                "category_label": TOPIC_LABELS_CLEAN.get(category, category),
                "interaction_log_odds": estimate,
                "std_error_quasi": standard_error,
                "z_value": z_value,
                "p_value": normal_p_value(z_value),
                "odds_ratio_per_month": math.exp(estimate),
                "odds_ratio_ci_lower": math.exp(estimate - 1.96 * standard_error),
                "odds_ratio_ci_upper": math.exp(estimate + 1.96 * standard_error),
                "n_cells": fit["n_obs"],
                "dispersion": fit["dispersion"],
            }
        )
    adjusted = bh_adjust([row["p_value"] for row in output_rows])
    for row, p_adjusted in zip(output_rows, adjusted):
        row["p_value_bh"] = p_adjusted
    return output_rows


def yearly_media_type_effects(temporal_cells: list[dict], family: str) -> list[dict]:
    totals: dict[tuple[str, str], float] = Counter()
    category_counts: dict[tuple[str, str, str], float] = Counter()
    categories = TOPIC_ORDER if family == "topic" else tuple(EMOTION_ORDER)
    for row in temporal_cells:
        year = row["month"][:4]
        media_type = row["media_type"]
        totals[(year, media_type)] += row["comment_count"]
        category_counts[(year, media_type, row["category"])] += row["successes"]
    years = sorted({year for year, _ in totals})
    rows = []
    for category in categories:
        for year in years:
            traditional_share = safe_ratio(category_counts[(year, "traditional", category)], totals[(year, "traditional")])
            emerging_share = safe_ratio(category_counts[(year, "emerging", category)], totals[(year, "emerging")])
            rows.append(
                {
                    "family": family,
                    "year": year,
                    "category": category,
                    "category_label": TOPIC_LABELS_CLEAN.get(category, category),
                    "traditional_share": traditional_share,
                    "emerging_share": emerging_share,
                    "traditional_minus_emerging": traditional_share - emerging_share,
                    "traditional_comment_count": totals[(year, "traditional")],
                    "emerging_comment_count": totals[(year, "emerging")],
                }
            )
    return rows


def channel_robustness(
    videos: dict[str, dict],
    channel_data: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    channels_by_media: dict[str, list[str]] = defaultdict(list)
    channel_views: Counter = Counter()
    channel_platform_comments: Counter = Counter()
    for video in videos.values():
        channel = video["channel"]
        channels_by_media[video["media_type"]].append(channel) if channel not in channels_by_media[video["media_type"]] else None
        channel_views[channel] += video["view_count"]
        channel_platform_comments[channel] += video["platform_comment_count"]

    channel_topic = channel_data["topic_counts"]
    channel_emotion = channel_data["emotion_counts"]
    channel_comment_counts = channel_data["comment_counts"]
    all_channels = sorted(set(channel_views) | set(channel_comment_counts))
    media_for_channel = {channel: media_type_for_channel(channel) for channel in all_channels}

    def effect_rows(category_names: tuple[str, ...], value_map: dict[str, np.ndarray], family: str) -> list[dict]:
        total_counts = {
            media_type: sum(channel_comment_counts[channel] for channel in all_channels if media_for_channel[channel] == media_type)
            for media_type in MEDIA_TYPES
        }
        total_values = {
            media_type: np.sum(
                [value_map[channel] for channel in all_channels if media_for_channel[channel] == media_type],
                axis=0,
            )
            for media_type in MEDIA_TYPES
        }
        rows = []
        for category_index, category in enumerate(category_names):
            full_effect = safe_ratio(total_values["traditional"][category_index], total_counts["traditional"]) - safe_ratio(total_values["emerging"][category_index], total_counts["emerging"])
            leave_one_out_values = []
            for channel in all_channels:
                media_type = media_for_channel[channel]
                numerator_traditional = total_values["traditional"][category_index] - (value_map[channel][category_index] if media_type == "traditional" else 0)
                numerator_emerging = total_values["emerging"][category_index] - (value_map[channel][category_index] if media_type == "emerging" else 0)
                denominator_traditional = total_counts["traditional"] - (channel_comment_counts[channel] if media_type == "traditional" else 0)
                denominator_emerging = total_counts["emerging"] - (channel_comment_counts[channel] if media_type == "emerging" else 0)
                leave_one_out_values.append(
                    safe_ratio(numerator_traditional, denominator_traditional) - safe_ratio(numerator_emerging, denominator_emerging)
                )
            channel_shares = {
                media_type: [
                    safe_ratio(value_map[channel][category_index], channel_comment_counts[channel])
                    for channel in all_channels
                    if media_for_channel[channel] == media_type and channel_comment_counts[channel]
                ]
                for media_type in MEDIA_TYPES
            }
            equal_channel_effect = float(np.mean(channel_shares["traditional"]) - np.mean(channel_shares["emerging"]))
            rows.append(
                {
                    "family": family,
                    "category": category,
                    "category_label": TOPIC_LABELS_CLEAN.get(category, category),
                    "full_sample_traditional_minus_emerging": full_effect,
                    "leave_one_out_min": min(leave_one_out_values),
                    "leave_one_out_max": max(leave_one_out_values),
                    "leave_one_out_sign_consistency": safe_ratio(
                        sum((value >= 0) == (full_effect >= 0) for value in leave_one_out_values),
                        len(leave_one_out_values),
                    ),
                    "equal_channel_weighted_traditional_minus_emerging": equal_channel_effect,
                    "traditional_channel_count": len(channel_shares["traditional"]),
                    "emerging_channel_count": len(channel_shares["emerging"]),
                }
            )
        return rows

    topic_rows = effect_rows(TOPIC_ORDER, channel_topic, "topic")
    emotion_rows = effect_rows(tuple(EMOTION_ORDER), channel_emotion, "emotion")

    engagement_rows = []
    for channel in all_channels:
        views = channel_views[channel]
        observed = channel_comment_counts[channel]
        platform_comments = channel_platform_comments[channel]
        engagement_rows.append(
            {
                "channel": channel,
                "channel_label": display_label_for_channel(channel),
                "media_type": media_for_channel[channel],
                "media_type_label": MEDIA_TYPE_LABELS[media_for_channel[channel]],
                "video_view_count": views,
                "platform_comment_count": platform_comments,
                "observed_comment_count": observed,
                "unique_observed_commenter_count": len(channel_data["users"].get(channel, set())),
                "observed_comments_per_1000_views": safe_ratio(observed * 1000, views),
                "unique_commenters_per_1000_views": safe_ratio(len(channel_data["users"].get(channel, set())) * 1000, views),
                "observed_comment_coverage_of_platform_count": safe_ratio(observed, platform_comments),
            }
        )
    return topic_rows, emotion_rows, engagement_rows


def channel_bootstrap_engagement(engagement_rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for metric, numerator_name in (
        ("Observed comments per 1,000 views", "observed_comment_count"),
        ("Unique commenters per 1,000 views", "unique_observed_commenter_count"),
    ):
        estimates = {}
        samples = {}
        for media_type in MEDIA_TYPES:
            selected = [row for row in engagement_rows if row["media_type"] == media_type]
            numerators = np.asarray([row[numerator_name] for row in selected], dtype=float)
            denominators = np.asarray([row["video_view_count"] for row in selected], dtype=float)
            estimates[media_type] = safe_ratio(numerators.sum() * 1000, denominators.sum())
            indices = rng.integers(0, len(selected), size=(2000, len(selected)))
            sample_numerators = numerators[indices].sum(axis=1)
            sample_denominators = denominators[indices].sum(axis=1)
            samples[media_type] = sample_numerators * 1000 / np.maximum(sample_denominators, 1.0)
            rows.append(
                {
                    "metric": metric,
                    "media_type": media_type,
                    "estimate_per_1000_views": estimates[media_type],
                    "ci_lower": float(np.quantile(samples[media_type], 0.025)),
                    "ci_upper": float(np.quantile(samples[media_type], 0.975)),
                    "bootstrap_unit": "channel",
                    "replicates": 2000,
                }
            )
        differences = samples["traditional"][:, None] - samples["emerging"][None, :]
        difference_samples = differences.ravel()
        rows.append(
            {
                "metric": metric,
                "media_type": "traditional_minus_emerging",
                "estimate_per_1000_views": estimates["traditional"] - estimates["emerging"],
                "ci_lower": float(np.quantile(difference_samples, 0.025)),
                "ci_upper": float(np.quantile(difference_samples, 0.975)),
                "bootstrap_unit": "channel",
                "replicates": 2000,
            }
        )
    return rows


def user_equal_weighted_sensitivity(month_users: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    """Compare comment-weighted and equal-user-month compositions."""
    topic_totals = {media_type: np.zeros(len(TOPIC_ORDER), dtype=float) for media_type in MEDIA_TYPES}
    emotion_totals = {media_type: np.zeros(len(EMOTION_ORDER), dtype=float) for media_type in MEDIA_TYPES}
    comment_totals = Counter()
    topic_user_mean_sum = {media_type: np.zeros(len(TOPIC_ORDER), dtype=float) for media_type in MEDIA_TYPES}
    emotion_user_mean_sum = {media_type: np.zeros(len(EMOTION_ORDER), dtype=float) for media_type in MEDIA_TYPES}
    user_month_counts = Counter()

    for users in month_users.values():
        for record in users.values():
            if not record.get("dynamic_tier"):
                continue
            for media_type in MEDIA_TYPES:
                by_media = record["by_media"][media_type]
                count = by_media["comment_count"]
                if not count:
                    continue
                comment_totals[media_type] += count
                topic_totals[media_type] += by_media["topic_counts"]
                emotion_totals[media_type] += by_media["emotion_counts"]
                topic_user_mean_sum[media_type] += by_media["topic_counts"] / count
                emotion_user_mean_sum[media_type] += by_media["emotion_counts"] / count
                user_month_counts[media_type] += 1

    def rows_for_family(names: tuple[str, ...], totals: dict, means: dict, family: str) -> list[dict]:
        output = []
        for index, category in enumerate(names):
            comment_weighted = {
                media_type: safe_ratio(totals[media_type][index], comment_totals[media_type])
                for media_type in MEDIA_TYPES
            }
            equal_user = {
                media_type: safe_ratio(means[media_type][index], user_month_counts[media_type])
                for media_type in MEDIA_TYPES
            }
            weighted_effect = comment_weighted["traditional"] - comment_weighted["emerging"]
            user_effect = equal_user["traditional"] - equal_user["emerging"]
            output.append(
                {
                    "family": family,
                    "category": category,
                    "category_label": TOPIC_LABELS_CLEAN.get(category, category),
                    "traditional_comment_weighted_share": comment_weighted["traditional"],
                    "emerging_comment_weighted_share": comment_weighted["emerging"],
                    "traditional_equal_user_month_share": equal_user["traditional"],
                    "emerging_equal_user_month_share": equal_user["emerging"],
                    "comment_weighted_traditional_minus_emerging": weighted_effect,
                    "equal_user_month_traditional_minus_emerging": user_effect,
                    "direction_agrees": (weighted_effect >= 0) == (user_effect >= 0),
                    "traditional_user_month_count": user_month_counts["traditional"],
                    "emerging_user_month_count": user_month_counts["emerging"],
                }
            )
        return output

    return (
        rows_for_family(TOPIC_ORDER, topic_totals, topic_user_mean_sum, "topic"),
        rows_for_family(tuple(EMOTION_ORDER), emotion_totals, emotion_user_mean_sum, "emotion"),
    )


def mobility_durability_and_cohorts(month_users: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Separate one-month absence, sustained absence, reactivation, and entry cohorts."""
    months = sorted(month_users)
    durability_rows = []
    cohort_rows = []
    timeline_rows = []
    ever_seen: set[str] = set()
    previous_active: set[str] = set()

    for month_index, month in enumerate(months):
        records = month_users[month]
        active_users = set(records)
        current_counts = Counter()
        media_comment_counts = Counter()
        media_active_users = {media_type: set() for media_type in MEDIA_TYPES}
        for user_id, record in records.items():
            current_counts[record["state"]] += 1
            for media_type in MEDIA_TYPES:
                count = record["by_media"][media_type]["comment_count"]
                media_comment_counts[media_type] += count
                if count:
                    media_active_users[media_type].add(user_id)
        cohort_rows.append(
            {
                "month": month,
                "active_user_count": len(active_users),
                "first_observed_user_count": len(active_users - ever_seen),
                "retained_from_previous_month_count": len(active_users & previous_active),
                "reactivated_user_count": len((active_users & ever_seen) - previous_active),
                "traditional_only_user_count": current_counts["traditional_only"],
                "emerging_only_user_count": current_counts["emerging_only"],
                "both_media_user_count": current_counts["both"],
            }
        )
        timeline_rows.append(
            {
                "month": month,
                "month_start": f"{month}-01",
                "observed_comment_count": sum(media_comment_counts.values()),
                "active_user_count": len(active_users),
                "traditional_comment_count": media_comment_counts["traditional"],
                "emerging_comment_count": media_comment_counts["emerging"],
                "traditional_active_user_count": len(media_active_users["traditional"]),
                "emerging_active_user_count": len(media_active_users["emerging"]),
                "candidate_comment_peak": False,
                "political_event_label": "",
                "event_date": "",
                "event_source_url": "",
                "event_source_note": "",
            }
        )
        ever_seen.update(active_users)
        previous_active = active_users

        if month_index >= len(months) - 1:
            continue
        for tier in TIER_ORDER:
            for from_state in STATE_ORDER[:-1]:
                selected = [
                    user_id for user_id, record in records.items()
                    if record.get("dynamic_tier") == tier and record["state"] == from_state
                ]
                if not selected:
                    continue
                next_month_users = set(month_users[months[month_index + 1]])
                one_month_absent = [user_id for user_id in selected if user_id not in next_month_users]
                common = {
                    "from_month": month,
                    "dynamic_tier": tier,
                    "from_state": from_state,
                    "from_state_total": len(selected),
                    "one_month_absent_count": len(one_month_absent),
                    "one_month_absent_rate": safe_ratio(len(one_month_absent), len(selected)),
                }
                row = dict(common)
                if month_index + 2 < len(months):
                    month_after_gap_users = set(month_users[months[month_index + 2]])
                    sustained_two_month = [user_id for user_id in one_month_absent if user_id not in month_after_gap_users]
                    row["two_month_absent_count"] = len(sustained_two_month)
                    row["two_month_absent_rate"] = safe_ratio(len(sustained_two_month), len(selected))
                    row["reactivated_after_one_month_gap_count"] = len(one_month_absent) - len(sustained_two_month)
                    row["reactivated_after_one_month_gap_rate"] = safe_ratio(len(one_month_absent) - len(sustained_two_month), len(selected))
                else:
                    row["two_month_absent_count"] = ""
                    row["two_month_absent_rate"] = ""
                    row["reactivated_after_one_month_gap_count"] = ""
                    row["reactivated_after_one_month_gap_rate"] = ""
                if month_index + 3 < len(months):
                    third_month_users = set(month_users[months[month_index + 3]])
                    sustained_three_month = [
                        user_id for user_id in one_month_absent
                        if user_id not in month_after_gap_users and user_id not in third_month_users
                    ]
                    row["three_month_absent_count"] = len(sustained_three_month)
                    row["three_month_absent_rate"] = safe_ratio(len(sustained_three_month), len(selected))
                else:
                    row["three_month_absent_count"] = ""
                    row["three_month_absent_rate"] = ""
                durability_rows.append(row)

    for index, row in enumerate(timeline_rows):
        total = row["observed_comment_count"]
        previous_total = timeline_rows[index - 1]["observed_comment_count"] if index else None
        next_total = timeline_rows[index + 1]["observed_comment_count"] if index + 1 < len(timeline_rows) else None
        row["candidate_comment_peak"] = bool(previous_total is not None and next_total is not None and total > previous_total and total > next_total)
    return durability_rows, cohort_rows, timeline_rows


def plot_temporal_interactions(topic_rows: list[dict], emotion_rows: list[dict], path: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.3), constrained_layout=True)
    for axis, rows, title in zip(axes, (topic_rows, emotion_rows), ("Topic prevalence", "Emotion prevalence")):
        sorted_rows = sorted(rows, key=lambda row: row["odds_ratio_per_month"])
        y_positions = np.arange(len(sorted_rows))
        estimates = np.asarray([row["odds_ratio_per_month"] for row in sorted_rows])
        lower = np.asarray([row["odds_ratio_ci_lower"] for row in sorted_rows])
        upper = np.asarray([row["odds_ratio_ci_upper"] for row in sorted_rows])
        axis.errorbar(estimates, y_positions, xerr=[estimates - lower, upper - estimates], fmt="o", color="#28666E", ecolor="#7A9E9F", capsize=3, markersize=5)
        axis.axvline(1, color="#4B5563", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([row["category_label"] for row in sorted_rows], fontsize=8.5)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Emerging-to-traditional change in odds per month")
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_channel_robustness(topic_rows: list[dict], emotion_rows: list[dict], path: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), constrained_layout=True)
    for axis, rows, title in zip(axes, (topic_rows, emotion_rows), ("Topic differences", "Emotion differences")):
        sorted_rows = sorted(rows, key=lambda row: row["full_sample_traditional_minus_emerging"])
        y_positions = np.arange(len(sorted_rows))
        full = np.asarray([row["full_sample_traditional_minus_emerging"] for row in sorted_rows])
        lower = np.asarray([row["leave_one_out_min"] for row in sorted_rows])
        upper = np.asarray([row["leave_one_out_max"] for row in sorted_rows])
        axis.hlines(y_positions, lower, upper, color="#9CA3AF", linewidth=2.2, zorder=1)
        axis.scatter(full, y_positions, s=28, color="#28666E", zorder=2)
        axis.axvline(0, color="#4B5563", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([row["category_label"] for row in sorted_rows], fontsize=8.5)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Traditional minus emerging comment-share difference")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_user_equal_weighted_sensitivity(topic_rows: list[dict], emotion_rows: list[dict], path: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), constrained_layout=True)
    for axis, rows, title in zip(axes, (topic_rows, emotion_rows), ("Topic composition", "Emotion composition")):
        selected = sorted(rows, key=lambda row: row["comment_weighted_traditional_minus_emerging"])
        y_positions = np.arange(len(selected))
        weighted = np.asarray([row["comment_weighted_traditional_minus_emerging"] for row in selected])
        user_weighted = np.asarray([row["equal_user_month_traditional_minus_emerging"] for row in selected])
        axis.hlines(y_positions, weighted, user_weighted, color="#B6C2D1", linewidth=1.6, zorder=1)
        axis.scatter(weighted, y_positions, s=32, color="#28666E", label="Comment weighted", zorder=2)
        axis.scatter(user_weighted, y_positions, s=32, color="#C58B00", label="Equal user-month", zorder=3)
        axis.axvline(0, color="#4B5563", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([row["category_label"] for row in selected], fontsize=8.5)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Traditional minus emerging share difference")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_mobility_durability(rows: list[dict], path: Path) -> None:
    configure_style()
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: Counter = Counter()
    for row in rows:
        month = row["from_month"]
        weight = float(row["from_state_total"])
        totals[month] += weight
        for metric in ("one_month_absent_count", "two_month_absent_count", "three_month_absent_count", "reactivated_after_one_month_gap_count"):
            value = row.get(metric)
            if value != "":
                grouped[month][metric] += float(value)
    months = sorted(grouped)
    fig, axis = plt.subplots(figsize=(11.6, 5.2), constrained_layout=True)
    lines = [
        ("one_month_absent_count", "Absent next month", "#4F83B9"),
        ("two_month_absent_count", "Absent for two months", "#28666E"),
        ("three_month_absent_count", "Absent for three months", "#6B7280"),
        ("reactivated_after_one_month_gap_count", "Reactivated after one-month gap", "#C58B00"),
    ]
    x_positions = np.arange(len(months))
    for metric, label, color in lines:
        values = [safe_ratio(grouped[month].get(metric, float("nan")), totals[month]) for month in months]
        axis.plot(x_positions, values, marker="o", markersize=3.8, linewidth=1.7, color=color, label=label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Share of active commenters at month t")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_mobility_durability_matrix(rows: list[dict], path: Path) -> None:
    configure_style()
    metrics = (
        ("one_month_absent_count", "Absent next month"),
        ("two_month_absent_count", "Absent for two months"),
        ("reactivated_after_one_month_gap_count", "Reactivated after one-month gap"),
    )
    months = sorted({row["from_month"] for row in rows})
    figure, axes_grid = plt.subplots(2, 2, figsize=(14.2, 8.0), constrained_layout=True)
    all_rates = []
    matrices = {}
    for tier in TIER_ORDER:
        matrix = np.full((len(metrics), len(months)), np.nan)
        for month_index, month in enumerate(months):
            selected = [row for row in rows if row["dynamic_tier"] == tier and row["from_month"] == month]
            denominator = sum(float(row["from_state_total"]) for row in selected)
            for metric_index, (metric, _) in enumerate(metrics):
                values = [row[metric] for row in selected if row.get(metric) != ""]
                if values and denominator:
                    matrix[metric_index, month_index] = sum(float(value) for value in values) / denominator
        matrices[tier] = matrix
        all_rates.extend(matrix[np.isfinite(matrix)].tolist())
    lower = min(all_rates) if all_rates else 0.0
    upper = max(all_rates) if all_rates else 1.0
    color_map = plt.colormaps["YlGnBu"].copy()
    color_map.set_bad("#E5E7EB")
    image = None
    for axis, tier in zip(axes_grid.ravel(), TIER_ORDER):
        image = axis.imshow(np.ma.masked_invalid(matrices[tier]), aspect="auto", cmap=color_map, vmin=lower, vmax=upper)
        axis.set_title(TIER_LABELS[tier], fontsize=10.5, pad=8)
        axis.set_xticks(np.arange(len(months)))
        axis.set_xticklabels(months, rotation=45, ha="right", fontsize=7.4)
        axis.set_yticks(np.arange(len(metrics)))
        axis.set_yticklabels([label for _, label in metrics], fontsize=8.3)
        axis.tick_params(length=0)
    colorbar = figure.colorbar(image, ax=axes_grid.ravel().tolist(), shrink=0.84, pad=0.02)
    colorbar.set_label("Share of active commenters at month t", fontsize=9)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def plot_yearly_effect_dumbbell(topic_rows: list[dict], emotion_rows: list[dict], path: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(14.3, 7.4), constrained_layout=True)
    colors = {"2024": "#28666E", "2025": "#C58B00"}
    for axis, rows, title in zip(axes, (topic_rows, emotion_rows), ("Topic composition", "Emotion composition")):
        by_category = defaultdict(dict)
        for row in rows:
            by_category[row["category"]][row["year"]] = row
        ordered = sorted(
            by_category,
            key=lambda category: by_category[category].get("2025", by_category[category].get("2024"))["traditional_minus_emerging"],
        )
        y_positions = np.arange(len(ordered))
        for y_position, category in zip(y_positions, ordered):
            entries = by_category[category]
            values = [entries[year]["traditional_minus_emerging"] for year in ("2024", "2025") if year in entries]
            if len(values) == 2:
                axis.hlines(y_position, values[0], values[1], color="#B6C2D1", linewidth=1.8, zorder=1)
            for year in ("2024", "2025"):
                if year in entries:
                    axis.scatter(entries[year]["traditional_minus_emerging"], y_position, s=38, color=colors[year], zorder=2, label=year if y_position == 0 else None)
        axis.axvline(0, color="#4B5563", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([by_category[category][next(iter(by_category[category]))]["category_label"] for category in ordered], fontsize=8.5)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Traditional minus emerging share difference")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def plot_engagement_bootstrap(rows: list[dict], path: Path) -> None:
    configure_style()
    plotted = [row for row in rows if row["media_type"] in MEDIA_TYPES]
    metrics = list(dict.fromkeys(row["metric"] for row in plotted))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 4.8), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    colors = {"traditional": "#4F83B9", "emerging": "#D95D5D"}
    for axis, metric in zip(axes, metrics):
        by_media = {row["media_type"]: row for row in plotted if row["metric"] == metric}
        selected = [by_media[media_type] for media_type in MEDIA_TYPES]
        x_positions = np.arange(len(selected))
        estimates = np.asarray([row["estimate_per_1000_views"] for row in selected])
        lower = np.asarray([row["ci_lower"] for row in selected])
        upper = np.asarray([row["ci_upper"] for row in selected])
        axis.errorbar(x_positions, estimates, yerr=[estimates - lower, upper - estimates], fmt="none", ecolor="#6B7280", capsize=4, linewidth=1.2)
        axis.scatter(x_positions, estimates, s=50, c=[colors[row["media_type"]] for row in selected], zorder=3)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(["Traditional", "Emerging"])
        axis.set_title(metric, fontsize=10, pad=9)
        axis.set_ylabel("Rate per 1,000 views")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_mobility_terms(rows: list[dict], path: Path, dynamic: bool) -> None:
    configure_style()
    if dynamic:
        keep_terms = [row for row in rows if row["term"].startswith("Dynamic tier:")]
        group_order = ["single_to_both", "direct_media_switch", "same_state_retention", "next_month_inactivity"]
        titles = ["Single to Both", "Direct switch", "Same-state retention", "Next-month inactivity"]
    else:
        keep_terms = [row for row in rows if row["term"].startswith("Topic:") or row["term"].startswith("Emotion:")]
        group_order = ["next_month_both_from_single", "next_month_inactivity"]
        titles = ["Next-month entry to Both", "Next-month inactivity"]
    if dynamic:
        fig, axes_grid = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
        axes = list(axes_grid.ravel())
    else:
        fig, axes_grid = plt.subplots(1, len(group_order), figsize=(14.2, 8.4), constrained_layout=True)
        axes = [axes_grid] if len(group_order) == 1 else list(axes_grid)
    for axis, group, title in zip(axes, group_order, titles):
        selected = [row for row in keep_terms if row["outcome"] == group]
        selected.sort(key=lambda row: row["odds_ratio"])
        y_positions = np.arange(len(selected))
        estimates = np.asarray([row["odds_ratio"] for row in selected])
        lower = np.asarray([row["odds_ratio_ci_lower"] for row in selected])
        upper = np.asarray([row["odds_ratio_ci_upper"] for row in selected])
        colors = ["#28666E" if row["term"].startswith("Topic:") else "#C58B00" for row in selected]
        if dynamic:
            colors = ["#28666E"] * len(selected)
        axis.errorbar(estimates, y_positions, xerr=[estimates - lower, upper - estimates], fmt="none", ecolor="#9CA3AF", capsize=3, zorder=1)
        axis.scatter(estimates, y_positions, c=colors, s=26, zorder=2)
        axis.axvline(1, color="#4B5563", linewidth=1)
        if dynamic:
            axis.set_xscale("log")
        axis.set_yticks(y_positions)
        labels = [row["term"].replace("Topic: ", "").replace("Emotion: ", "") for row in selected]
        axis.set_yticklabels(labels, fontsize=8.2)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Odds ratio (95% confidence interval)")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def summarize_dynamic_tier_mobility_rates(
    rows: list[dict],
    seed: int,
    replicates: int = 4000,
) -> list[dict]:
    """Estimate descriptive tier-specific transition rates with month-pair CIs."""
    outcomes = (
        "single_to_both",
        "direct_media_switch",
        "same_state_retention",
        "next_month_inactivity",
    )
    pairs = sorted({row["month_pair"] for row in rows})
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    grouped: dict[tuple[str, str], np.ndarray] = {}
    for tier in TIER_ORDER:
        for outcome in outcomes:
            grouped[(tier, outcome)] = np.zeros((len(pairs), 2), dtype=float)

    for row in rows:
        key = (row["dynamic_tier"], row["outcome"])
        if key not in grouped:
            continue
        index = pair_index[row["month_pair"]]
        grouped[key][index, 0] += float(row["successes"])
        grouped[key][index, 1] += float(row["trials"])

    rng = np.random.default_rng(seed)
    output_rows = []
    for tier in TIER_ORDER:
        for outcome in outcomes:
            values = grouped[(tier, outcome)]
            successes = values[:, 0]
            trials = values[:, 1]
            observed_trials = trials.sum()
            if not observed_trials:
                continue
            sample_indices = rng.integers(0, len(pairs), size=(replicates, len(pairs)))
            sampled_successes = successes[sample_indices].sum(axis=1)
            sampled_trials = trials[sample_indices].sum(axis=1)
            sampled_rates = np.divide(
                sampled_successes,
                sampled_trials,
                out=np.full(replicates, np.nan),
                where=sampled_trials > 0,
            )
            denominator = (
                "single-media commenters"
                if outcome in {"single_to_both", "direct_media_switch"}
                else "all active commenters"
            )
            output_rows.append(
                {
                    "dynamic_tier": tier,
                    "dynamic_tier_label": TIER_LABELS[tier],
                    "outcome": outcome,
                    "observed_rate": float(successes.sum() / observed_trials),
                    "ci_lower": float(np.nanpercentile(sampled_rates, 2.5)),
                    "ci_upper": float(np.nanpercentile(sampled_rates, 97.5)),
                    "successes": int(successes.sum()),
                    "trials": int(observed_trials),
                    "bootstrap_unit": "adjacent-month pair",
                    "bootstrap_replicates": replicates,
                    "denominator": denominator,
                }
            )
    return output_rows


def plot_dynamic_tier_mobility_profile(rows: list[dict], path: Path) -> None:
    """Plot descriptive rates, avoiding the hard-to-read multi-panel OR forest plot."""
    configure_style()
    outcome_specs = (
        ("single_to_both", "Entry into both media types", "From single-media commenters"),
        ("direct_media_switch", "Direct media-type switch", "From single-media commenters"),
        ("same_state_retention", "Same-state retention", "From all active commenters"),
        ("next_month_inactivity", "Next-month non-observation", "From all active commenters"),
    )
    lookup = {(row["dynamic_tier"], row["outcome"]): row for row in rows}
    figure, axes_grid = plt.subplots(2, 2, figsize=(12.4, 7.0), sharey=True, constrained_layout=True)
    for axis, (outcome, title, denominator) in zip(axes_grid.ravel(), outcome_specs):
        selected = [lookup[(tier, outcome)] for tier in TIER_ORDER if (tier, outcome) in lookup]
        y_positions = np.arange(len(selected))
        rates = np.asarray([row["observed_rate"] for row in selected])
        lower = np.asarray([row["ci_lower"] for row in selected])
        upper = np.asarray([row["ci_upper"] for row in selected])
        axis.errorbar(
            rates,
            y_positions,
            xerr=[rates - lower, upper - rates],
            fmt="none",
            ecolor="#94A3B8",
            elinewidth=1.6,
            capsize=3.5,
            zorder=1,
        )
        axis.scatter(rates, y_positions, s=56, color="#147D8A", edgecolor="white", linewidth=0.8, zorder=2)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([row["dynamic_tier_label"] for row in selected], fontsize=9.4)
        axis.invert_yaxis()
        maximum = max(float(upper.max()) * 1.12, 0.025)
        axis.set_xlim(0, min(1.0, maximum))
        decimals = 1 if outcome == "direct_media_switch" else 0
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))
        axis.set_title(title, fontsize=11.1, pad=12)
        axis.set_xlabel("Observed adjacent-month rate\n(95% descriptive percentile interval)", fontsize=8.6)
        axis.grid(axis="x", color="#E2E8F0", linewidth=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def generate_corpus_validation_sample(videos: dict[str, dict], capacity: int, rng: random.Random) -> list[dict]:
    reservoirs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: Counter = Counter()
    for video in videos.values():
        stratum = (video["media_type"], quarter_from_month(video["published_month"]))
        seen[stratum] += 1
        reservoir_add(reservoirs[stratum], seen[stratum], video, capacity, rng)
    rows = []
    sample_index = 1
    for (media_type, quarter), selected in sorted(reservoirs.items()):
        for video in selected:
            title = video["title"]
            description = video["description"]
            rows.append(
                {
                    "sample_id": f"CORPUS-{sample_index:04d}",
                    "sampling_stratum": f"{MEDIA_TYPE_LABELS[media_type]} | {quarter}",
                    "video_id": video["video_id"],
                    "media_type": media_type,
                    "channel_label": video["channel_label"],
                    "published_at": video["published_at"],
                    "matched_keywords": video["matched_keywords"],
                    "title": title,
                    "description": description,
                    "keyword_in_title": any(keyword in title for keyword in ("罷免", "二階罷免", "大罷免")),
                    "keyword_in_description": any(keyword in description for keyword in ("罷免", "二階罷免", "大罷免")),
                    "manual_event_relevant": "",
                    "manual_relevance_basis": "",
                    "annotator_id": "",
                    "annotation_date": "",
                    "second_manual_event_relevant": "",
                    "second_annotator_id": "",
                    "second_annotation_date": "",
                    "notes": "",
                }
            )
            sample_index += 1
    return rows


def generate_topic_validation_sample(topic_reservoirs: dict[tuple[str, str], list[dict]]) -> list[dict]:
    rows = []
    sample_index = 1
    for (media_type, auto_primary_topic), selected in sorted(topic_reservoirs.items()):
        for candidate in selected:
            row = {
                "sample_id": f"TOPIC-{sample_index:04d}",
                "sampling_stratum": f"{MEDIA_TYPE_LABELS[media_type]} | auto-primary: {auto_primary_topic}",
                **candidate,
                "manual_any_topic": "",
                "annotator_id": "",
                "annotation_date": "",
                "second_manual_any_topic": "",
                "second_annotator_id": "",
                "second_annotation_date": "",
                "notes": "",
            }
            for topic in TOPIC_ORDER:
                row[f"manual_{topic}"] = ""
                row[f"second_manual_{topic}"] = ""
            rows.append(row)
            sample_index += 1
    return rows


def write_validation_guide(path: Path) -> None:
    topic_lines = "\n".join(
        f"- `{topic}`: mark 1 only when the comment substantively invokes this domain; mere incidental word overlap should be 0. Automatic cue terms: {', '.join(keywords)}."
        for topic, keywords in TOPICS.items()
    )
    text = f"""# Manual Validation Guide

These templates support transparent measurement checks for the YouTube study.

## 1. Corpus relevance

For `manual_event_relevant`, enter `1` when the video is substantively about the focal Taiwan political event, `0` when it merely contains a generic or unrelated use of a query term, and `uncertain` when title and description do not permit a decision. Use `manual_relevance_basis` to note the contextual phrase or external cue used for the judgement. Do not silently remove uncertain cases.

Report the proportion of relevant videos among completed, non-uncertain rows separately by media type and time stratum. The template is stratified by media type and quarter so the final estimate should be reported with its sampling design. When a second coder is available, use `second_manual_event_relevant` and record the coder identifier separately.

## 2. Topic validation

Topic coding is multi-label: one comment may receive more than one `manual_<topic>` value of `1`. Use `0` when a topic is absent and `uncertain` only when interpretation is genuinely indeterminate. `manual_any_topic` should be `1` if at least one topic is present.

{topic_lines}

The sample is stratified by automatically assigned primary topic and media type. Report precision for each automatic topic within its sampled stratum. A second independent annotator should code a subset (at least 100 comments) in the `second_manual_*` columns so that agreement, preferably Cohen's kappa per label or percent agreement where prevalence is sparse, can be reported.

## 3. What this validation can support

It supports statements about keyword-query relevance and the accuracy of the topic dictionary as an observational measurement instrument. It does not turn the media-type comparison into a causal estimate. In the paper, call the human-coded labels a validation sample and retain the original automatic labels for reproducibility.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def copy_supplementary(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = destination / source.name
        shutil.copy2(source, target)
        print(f"Copied {target}")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    excluded_months = set(args.exclude_month or [])

    videos = load_videos(args.videos_json)
    print(f"Loaded {len(videos):,} in-scope videos")
    (
        month_users,
        channel_data,
        topic_reservoirs,
        topic_reservoir_seen,
        skipped_dict,
        _,
    ) = load_comment_data(
        args.comments_jsonl,
        videos,
        args.max_month,
        excluded_months,
        args.topic_sample_per_stratum,
        rng,
    )
    print(f"Loaded {sum(len(users) for users in month_users.values()):,} user-month records")

    month_users, dynamic_state_rows, dynamic_data = finalize_dynamic_tiers(month_users)
    transition_rows, transition_model_rows = build_dynamic_transition_rows(month_users)
    dynamic_model_rows = fit_dynamic_transition_models(transition_model_rows)
    dynamic_rate_rows = summarize_dynamic_tier_mobility_rates(transition_model_rows, args.seed)
    linkage_records = build_user_month_linkage_records(month_users)
    linkage_model_rows = fit_linkage_models(linkage_records)
    topic_temporal_cells, emotion_temporal_cells = build_temporal_cells(dynamic_data["temporal_counts"])
    topic_trend_rows = fit_trend_interactions(topic_temporal_cells, "topic")
    emotion_trend_rows = fit_trend_interactions(emotion_temporal_cells, "emotion")
    yearly_topic_rows = yearly_media_type_effects(topic_temporal_cells, "topic")
    yearly_emotion_rows = yearly_media_type_effects(emotion_temporal_cells, "emotion")
    channel_topic_rows, channel_emotion_rows, engagement_rows = channel_robustness(videos, channel_data)
    engagement_bootstrap_rows = channel_bootstrap_engagement(engagement_rows, args.seed)
    topic_user_sensitivity_rows, emotion_user_sensitivity_rows = user_equal_weighted_sensitivity(month_users)
    durability_rows, cohort_rows, timeline_rows = mobility_durability_and_cohorts(month_users)

    dynamic_dir = output_dir / "dynamic_tier"
    linkage_dir = output_dir / "user_month_linkage"
    temporal_dir = output_dir / "temporal_interactions"
    robustness_dir = output_dir / "channel_robustness"
    engagement_dir = output_dir / "engagement"
    user_sensitivity_dir = output_dir / "user_equal_weighted_sensitivity"
    durability_dir = output_dir / "mobility_durability"
    timeline_dir = output_dir / "event_timeline"
    validation_dir = output_dir / "validation"

    write_csv(
        dynamic_dir / "dynamic_tier_monthly_user_state_summary.csv",
        dynamic_state_rows,
        [
            "month", "user_id", "dynamic_tier", "dynamic_tier_label", "cumulative_comment_count",
            "month_comment_count", "state", "state_label", "dominant_topic", "dominant_emotion",
        ],
    )
    write_csv(
        dynamic_dir / "dynamic_tier_monthly_transition_summary.csv",
        transition_rows,
        [
            "from_month", "to_month", "month_pair", "dynamic_tier", "dynamic_tier_label",
            "from_state", "from_state_label", "to_state", "to_state_label", "transition_count",
            "from_state_total", "transition_rate",
        ],
    )
    write_csv(
        dynamic_dir / "dynamic_tier_transition_model_terms.csv",
        dynamic_model_rows,
        [
            "model", "outcome", "term", "estimate_log_odds", "std_error", "z_value", "p_value",
            "p_value_bh", "odds_ratio", "odds_ratio_ci_lower", "odds_ratio_ci_upper", "n_obs",
            "dispersion", "total_trials", "total_successes",
        ],
    )
    dynamic_plot = dynamic_dir / "dynamic_tier_mobility_odds_ratios.png"
    plot_mobility_terms(dynamic_model_rows, dynamic_plot, dynamic=True)
    dynamic_profile_path = dynamic_dir / "dynamic_tier_mobility_profile.png"
    plot_dynamic_tier_mobility_profile(dynamic_rate_rows, dynamic_profile_path)
    dynamic_rate_path = dynamic_dir / "dynamic_tier_mobility_rate_bootstrap_ci.csv"
    write_csv(
        dynamic_rate_path,
        dynamic_rate_rows,
        [
            "dynamic_tier", "dynamic_tier_label", "outcome", "observed_rate", "ci_lower", "ci_upper",
            "successes", "trials", "bootstrap_unit", "bootstrap_replicates", "denominator",
        ],
    )

    write_csv(
        linkage_dir / "user_month_discourse_to_mobility_model_terms.csv",
        linkage_model_rows,
        [
            "model", "outcome", "term", "estimate_log_odds", "std_error", "z_value", "p_value",
            "p_value_bh", "odds_ratio", "odds_ratio_ci_lower", "odds_ratio_ci_upper", "n_obs",
            "cluster_count", "outcome_rate",
        ],
    )
    linkage_plot = linkage_dir / "user_month_discourse_to_mobility_odds_ratios.png"
    plot_mobility_terms(linkage_model_rows, linkage_plot, dynamic=False)

    trend_fields = [
        "family", "category", "category_label", "interaction_log_odds", "std_error_quasi", "z_value",
        "p_value", "p_value_bh", "odds_ratio_per_month", "odds_ratio_ci_lower", "odds_ratio_ci_upper",
        "n_cells", "dispersion",
    ]
    write_csv(temporal_dir / "dynamic_tier_topic_monthly_trend_interactions.csv", topic_trend_rows, trend_fields)
    write_csv(temporal_dir / "dynamic_tier_emotion_monthly_trend_interactions.csv", emotion_trend_rows, trend_fields)
    temporal_plot = temporal_dir / "media_type_monthly_trend_interactions.png"
    plot_temporal_interactions(topic_trend_rows, emotion_trend_rows, temporal_plot)
    yearly_effect_fields = [
        "family", "year", "category", "category_label", "traditional_share", "emerging_share",
        "traditional_minus_emerging", "traditional_comment_count", "emerging_comment_count",
    ]
    write_csv(temporal_dir / "yearly_topic_media_type_effects.csv", yearly_topic_rows, yearly_effect_fields)
    write_csv(temporal_dir / "yearly_emotion_media_type_effects.csv", yearly_emotion_rows, yearly_effect_fields)
    yearly_dumbbell_plot = temporal_dir / "yearly_media_type_effect_dumbbell.png"
    plot_yearly_effect_dumbbell(yearly_topic_rows, yearly_emotion_rows, yearly_dumbbell_plot)

    channel_fields = [
        "family", "category", "category_label", "full_sample_traditional_minus_emerging", "leave_one_out_min",
        "leave_one_out_max", "leave_one_out_sign_consistency", "equal_channel_weighted_traditional_minus_emerging",
        "traditional_channel_count", "emerging_channel_count",
    ]
    write_csv(robustness_dir / "topic_leave_one_channel_out.csv", channel_topic_rows, channel_fields)
    write_csv(robustness_dir / "emotion_leave_one_channel_out.csv", channel_emotion_rows, channel_fields)
    robustness_plot = robustness_dir / "channel_leave_one_out_ranges.png"
    plot_channel_robustness(channel_topic_rows, channel_emotion_rows, robustness_plot)

    user_sensitivity_fields = [
        "family", "category", "category_label", "traditional_comment_weighted_share",
        "emerging_comment_weighted_share", "traditional_equal_user_month_share",
        "emerging_equal_user_month_share", "comment_weighted_traditional_minus_emerging",
        "equal_user_month_traditional_minus_emerging", "direction_agrees", "traditional_user_month_count",
        "emerging_user_month_count",
    ]
    write_csv(user_sensitivity_dir / "topic_user_equal_weighted_sensitivity.csv", topic_user_sensitivity_rows, user_sensitivity_fields)
    write_csv(user_sensitivity_dir / "emotion_user_equal_weighted_sensitivity.csv", emotion_user_sensitivity_rows, user_sensitivity_fields)
    user_sensitivity_plot = user_sensitivity_dir / "comment_weighted_vs_equal_user_composition.png"
    plot_user_equal_weighted_sensitivity(topic_user_sensitivity_rows, emotion_user_sensitivity_rows, user_sensitivity_plot)

    engagement_fields = [
        "channel", "channel_label", "media_type", "media_type_label", "video_view_count", "platform_comment_count",
        "observed_comment_count", "unique_observed_commenter_count", "observed_comments_per_1000_views",
        "unique_commenters_per_1000_views", "observed_comment_coverage_of_platform_count",
    ]
    write_csv(engagement_dir / "channel_engagement_rates.csv", engagement_rows, engagement_fields)
    write_csv(
        engagement_dir / "media_type_channel_bootstrap_engagement_ci.csv",
        engagement_bootstrap_rows,
        ["metric", "media_type", "estimate_per_1000_views", "ci_lower", "ci_upper", "bootstrap_unit", "replicates"],
    )
    engagement_plot = engagement_dir / "media_type_engagement_bootstrap_ci.png"
    plot_engagement_bootstrap(engagement_bootstrap_rows, engagement_plot)

    durability_fields = [
        "from_month", "dynamic_tier", "from_state", "from_state_total", "one_month_absent_count",
        "one_month_absent_rate", "two_month_absent_count", "two_month_absent_rate",
        "reactivated_after_one_month_gap_count", "reactivated_after_one_month_gap_rate",
        "three_month_absent_count", "three_month_absent_rate",
    ]
    write_csv(durability_dir / "mobility_durability_by_state_and_tier.csv", durability_rows, durability_fields)
    write_csv(
        durability_dir / "monthly_user_entry_return_retention_summary.csv",
        cohort_rows,
        [
            "month", "active_user_count", "first_observed_user_count", "retained_from_previous_month_count",
            "reactivated_user_count", "traditional_only_user_count", "emerging_only_user_count", "both_media_user_count",
        ],
    )
    durability_plot = durability_dir / "mobility_durability_by_horizon.png"
    plot_mobility_durability(durability_rows, durability_plot)
    durability_matrix_plot = durability_dir / "mobility_durability_tile_matrix.png"
    plot_mobility_durability_matrix(durability_rows, durability_matrix_plot)

    write_csv(
        timeline_dir / "monthly_activity_event_timeline_template.csv",
        timeline_rows,
        [
            "month", "month_start", "observed_comment_count", "active_user_count", "traditional_comment_count",
            "emerging_comment_count", "traditional_active_user_count", "emerging_active_user_count",
            "candidate_comment_peak", "political_event_label", "event_date", "event_source_url", "event_source_note",
        ],
    )

    corpus_rows = generate_corpus_validation_sample(videos, args.corpus_sample_per_stratum, rng)
    topic_rows = generate_topic_validation_sample(topic_reservoirs)
    corpus_fields = [
        "sample_id", "sampling_stratum", "video_id", "media_type", "channel_label", "published_at",
        "matched_keywords", "title", "description", "keyword_in_title", "keyword_in_description",
        "manual_event_relevant", "manual_relevance_basis", "annotator_id", "annotation_date",
        "second_manual_event_relevant", "second_annotator_id", "second_annotation_date", "notes",
    ]
    topic_fields = [
        "sample_id", "sampling_stratum", "comment_id", "video_id", "month", "media_type", "channel",
        "channel_label", "comment_text", "auto_topic_labels", "auto_primary_topic", "auto_emotion",
        "manual_any_topic", *[f"manual_{topic}" for topic in TOPIC_ORDER], "annotator_id", "annotation_date",
        "second_manual_any_topic", *[f"second_manual_{topic}" for topic in TOPIC_ORDER],
        "second_annotator_id", "second_annotation_date", "notes",
    ]
    write_csv(validation_dir / "corpus_relevance_validation_sample.csv", corpus_rows, corpus_fields)
    write_csv(validation_dir / "topic_multilabel_validation_sample.csv", topic_rows, topic_fields)
    write_validation_guide(validation_dir / "VALIDATION_GUIDE.md")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "strengthen_core_analyses.py",
        "scope": "Observational analyses of commenting activity; no causal media-effect interpretation.",
        "time_range": {"max_month": args.max_month, "excluded_months": sorted(excluded_months)},
        "dynamic_tier_definition": "Tier is assigned from cumulative in-scope comments through the current month only; user-months with fewer than two cumulative comments are excluded from tiered models.",
        "comment_loading": {
            "in_scope_videos": len(videos),
            "user_month_records": sum(len(users) for users in month_users.values()),
            "skipped": skipped_dict,
            "dynamic_tier_summary": dynamic_data["summary"],
        },
        "models": {
            "dynamic_transition": "Grouped quasi-binomial logit: source media state, dynamic tier, and monthly time index.",
            "discourse_to_mobility": "User-month logit with sandwich covariance clustered by user: current source state, dynamic tier, current-month activity, dominant topic, dominant emotion, and time.",
            "temporal_interactions": "Category-specific grouped quasi-binomial logit: media type, dynamic tier, month index, and media-type-by-month interaction.",
        },
        "validation_templates": {
            "corpus_relevance_rows": len(corpus_rows),
            "topic_multilabel_rows": len(topic_rows),
            "topic_reservoir_seen_by_stratum": {f"{key[0]} | {key[1]}": value for key, value in sorted(topic_reservoir_seen.items())},
        },
        "outputs": {
            "dynamic_tier": str(dynamic_dir),
            "linkage": str(linkage_dir),
            "temporal_interactions": str(temporal_dir),
            "channel_robustness": str(robustness_dir),
            "engagement": str(engagement_dir),
            "user_equal_weighted_sensitivity": str(user_sensitivity_dir),
            "mobility_durability": str(durability_dir),
            "event_timeline_template": str(timeline_dir),
            "validation": str(validation_dir),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    copy_supplementary(
        [
            dynamic_plot,
            dynamic_profile_path,
            dynamic_rate_path,
            linkage_plot,
            temporal_plot,
            yearly_dumbbell_plot,
            robustness_plot,
            user_sensitivity_plot,
            engagement_plot,
            durability_plot,
            durability_matrix_plot,
            summary_path,
        ],
        args.paper_supplementary_dir,
    )


if __name__ == "__main__":
    main()
