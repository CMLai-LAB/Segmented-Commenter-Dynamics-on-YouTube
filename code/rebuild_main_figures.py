#!/usr/bin/env python3
"""Rebuild the ten Results figures from existing aggregate CSV files."""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

import plot_channel_level_by_media_type as channels
import plot_epj_main_figures as topics
import plot_paper_statistical_figures as statistics
import plot_quarterly_media_type_transition_matrix_panels as transitions
import strengthen_core_analyses as mobility


ROOT = Path(__file__).resolve().parent


def reset_style(module):
    plt.rcdefaults()
    module.configure_style()


def run_plot(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT,
                        help="Workspace data directory or extracted google_drive_data directory.")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="Directory for regenerated plots, metadata, and figures/.")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()

    def input_dir(workspace_path, package_path):
        for candidate in (data_dir / workspace_path, data_dir / package_path):
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"Missing aggregate directory: {workspace_path} or {package_path} under {data_dir}")

    topic_input = input_dir("topic_by_media_type", "topic/topic_by_media_type")
    emotion_input = input_dir("emotion_by_media_type", "emotion/emotion_by_media_type")
    residual_input = input_dir("emotion_stat_tests_media_type_by_tier", "emotion/emotion_stat_tests_media_type_by_tier")
    transition_input = input_dir("user_mobility_by_media_type", "mobility/user_mobility_by_media_type")
    ci_input = input_dir("user_mobility_bootstrap_ci", "mobility/user_mobility_bootstrap_ci")
    dynamic_input = input_dir("analysis_strengthening/dynamic_tier", "mobility/dynamic_tier")
    selected_dir = output_root / "paper_selected_figures" / "main"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ("01_topic", "02_emotion", "03_user_mobility"):
        (selected_dir / name).mkdir(parents=True, exist_ok=True)
    topic_dir = output_root / "topic_by_media_type"
    emotion_dir = output_root / "emotion_by_media_type"
    statistical_dir = output_root / "paper_statistical_figures"
    for directory in (topic_dir, emotion_dir, statistical_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # These entry points only read aggregate CSVs and write plots and metadata.
    run_plot("plot_topic_intensity_difference_by_media_type.py", "--input-csv", str(topic_input / "traditional_vs_emerging_topic_comparison.csv"), "--output-dir", str(topic_dir))
    run_plot("plot_emotion_intensity_per_1000_views_by_media_type.py", "--input-csv", str(emotion_input / "media_type_emotion_summary.csv"), "--output-dir", str(emotion_dir))

    reset_style(topics)
    topics.plot_topic_difference_panels(topic_input / "media_type_quarter_topic_summary.csv", selected_dir)
    topics.plot_topic_tier_difference_dotplot(topic_input / "traditional_vs_emerging_topic_comparison.csv", selected_dir)

    reset_style(channels)
    channel_rows = channels.read_rows(topic_input / "channel_topic_summary.csv")
    channels.draw_heatmap(
        channel_rows,
        topic_dir / "channel_topic_share_heatmap.png",
        topic_dir / "channel_topic_share_heatmap_summary.json",
        item_key="topic", item_order=list(channels.TOPICS),
        item_labels=channels.TOPIC_LABELS_EN, metric="topic_comment_share",
        title="Channel-level lexical-domain presence shares",
        colorbar_label="Share of analyzed comments", cmap="YlGnBu", clip=False,
    )
    channels.draw_support_dotplot(
        channel_rows,
        channels.read_rows(topic_input / "channel_topic_media_type_support.csv"),
        topic_dir / "channel_topic_media_type_support_dotplot.png",
        topic_dir / "channel_topic_media_type_support_dotplot_summary.json",
        item_key="topic", item_labels=channels.TOPIC_LABELS_EN,
        metric="topic_comments_per_1000_views",
        title="Channel-Level Lexical-Domain Intensity",
        xlabel="Lexical-domain comments per 1,000 views",
    )

    reset_style(statistics)
    statistics.draw_emotion_residual_bar_panels(
        rows=statistics.read_csv(residual_input / "standardized_residuals.csv"),
        output_png=statistical_dir / "emotion_standardized_residual_diverging_bars.png",
        output_json=statistical_dir / "emotion_standardized_residual_diverging_bars_summary.json",
    )
    statistics.draw_media_mobility_ci(statistics.read_csv(ci_input / "media_type_mobility_bootstrap_ci.csv"), statistical_dir)

    reset_style(transitions)
    matrix_png = statistical_dir / "quarterly_media_type_transition_matrix_panels_clean.png"
    transitions.draw_matrix_panels(
        transitions.read_pairs(transition_input / "quarterly_user_transition_churn_summary.csv"), transition_input / "quarterly_user_transition_churn_summary.csv",
        matrix_png, matrix_png.with_name(matrix_png.stem + "_summary.json"),
        annotate=False, annotate_threshold=0.05, show_metrics=False,
    )

    reset_style(mobility)
    dynamic_dir = output_root / "analysis_strengthening" / "dynamic_tier"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    rates = statistics.read_csv(dynamic_input / "dynamic_tier_mobility_rate_bootstrap_ci.csv")
    for row in rates:
        for key in ("observed_rate", "ci_lower", "ci_upper"):
            row[key] = float(row[key])
    mobility.plot_dynamic_tier_mobility_profile(rates, dynamic_dir / "dynamic_tier_mobility_profile.png")

    # Copy the exact generated files to both the selected set and TeX inputs.
    outputs = [
        (topic_dir / "topic_intensity_difference_by_media_type.png", "01_topic", "Fig_01_topic_media_type_difference.png"),
        (selected_dir / "01_topic/Fig_03_quarterly_topic_media_type_difference_panels.png", "01_topic", "Fig_03_quarterly_topic_media_type_difference_panels.png"),
        (topic_dir / "channel_topic_share_heatmap.png", "01_topic", "Fig_04a_channel_topic_share_heatmap.png"),
        (topic_dir / "channel_topic_media_type_support_dotplot.png", "01_topic", "Fig_04b_channel_topic_media_type_support_dotplot.png"),
        (emotion_dir / "emotion_intensity_per_1000_views_by_media_type.png", "02_emotion", "Fig_06_emotion_intensity_per_1000_views.png"),
        (statistical_dir / "emotion_standardized_residual_diverging_bars.png", "02_emotion", "Fig_07_emotion_standardized_residual_diverging_bars.png"),
        (selected_dir / "01_topic/Fig_05_topic_tier_media_type_difference_dotplot.png", "01_topic", "Fig_05_topic_tier_media_type_difference_dotplot.png"),
        (matrix_png, "03_user_mobility", "Fig_09_quarterly_media_type_transition_matrix_panels_clean.png"),
        (statistical_dir / "media_type_mobility_bootstrap_ci.png", "03_user_mobility", "Fig_10_media_type_mobility_bootstrap_ci.png"),
        (dynamic_dir / "dynamic_tier_mobility_profile.png", "03_user_mobility", "Fig_11a_dynamic_tier_mobility_profile.png"),
    ]
    for source, folder, name in outputs:
        selected = selected_dir / folder / name
        if source != selected:
            shutil.copy2(source, selected)
            metadata = source.with_name(source.stem + "_summary.json")
            if metadata.exists():
                shutil.copy2(metadata, selected.with_name(selected.stem + "_summary.json"))
        shutil.copy2(selected, figures_dir / name)
        print(f"Synced {name}")


if __name__ == "__main__":
    main()
