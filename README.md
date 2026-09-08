# Segmented Commenter Dynamics on YouTube: Analysis Code

This repository contains the analysis and figure-generation code for the manuscript, "Segmented Commenter Dynamics on YouTube: Topic, Emotion, and Mobility Across Traditional and Emerging Media."

## Repository contents

- `code/`: scripts for keyword retrieval, topic and emotion summaries, statistical tests, mobility analysis, bootstrap intervals, emotion-agreement summaries, and final figures.
- `requirements.txt`: Python packages required by the included scripts.

## Data access

Non-identifying aggregate data supporting the reported tables, figures, and supplementary model summaries are available from [Google Drive](https://drive.google.com/file/d/1-Dp1QFtfk97BCR3n3nQwPRiFbm78EpDy/view?usp=sharing).

The data package excludes raw comment text, public author-channel identifiers, commenter-level linkage files, raw validation records, and user-level network nodes or edges. Some scripts document the full workflow and therefore require access to the non-deposited raw platform snapshots to be run from the first collection step.

## Software and model details

The released analysis and plotting scripts were checked with Python 3.12, NumPy 2.3.3, and Matplotlib 3.10.7. The emotion-validation script additionally requires PyTorch and Hugging Face Transformers; it uses the model identifier `Johnson8187/Chinese-Emotion` and the default model revision resolved by Hugging Face at execution time.

The dictionary-construction audit used CKIP Transformers for a sequential word-segmentation and part-of-speech-tagging workflow:

- Word segmentation: `ckiplab/bert-base-chinese-ws`
- Part-of-speech tagging: `ckiplab/bert-base-chinese-pos`

The CKIP model identifiers and their access date are documented in the manuscript. The original environment used for model inference was not preserved as a version-locked environment; the repository therefore does not claim unverified package versions or immutable model revisions.

## Analysis workflow

### Rebuild the ten Results figures from the released data

Extract `Youtube_recall_aggregate_data.zip` into this repository root so that
`google_drive_data/` sits alongside `code/`. With NumPy and Matplotlib installed,
run:

```bash
python3 code/rebuild_main_figures.py --data-dir google_drive_data --output-root reproduced
```

The final PNGs are written to `reproduced/figures/`, with matching selected
copies and plotting metadata under `reproduced/`. This command uses only the
released aggregate CSVs. It does not rerun raw-comment analysis, emotion
inference, or bootstrap estimation.

The release includes the dynamic-tier mobility rates, grouped transition
counts, and supplementary model terms in `google_drive_data/mobility/dynamic_tier/`.
The data-package README documents analysis populations, pooled Jaccard
estimation, and the model-specific BH correction families for Table S4.

### Full analysis workflow

The scripts correspond to the following production sequence. Running the
complete workflow from the first step requires the non-deposited raw platform
snapshots; the Google Drive package provides the resulting non-identifying
aggregate outputs.

1. `filter_by_keywords.py` applies the three recall-related retrieval terms.
2. `analyze_topic_by_media_type.py` and
   `analyze_topic_stat_tests_by_media_type.py` produce topic summaries and
   statistical-test outputs in their respective analysis output directories.
3. `analyze_emotion_by_media_type.py` and
   `analyze_emotion_media_type_by_tier_chi_square.py` produce the final emotion
   summaries and overall/tier-specific statistical tests. The older
   `analyze_emotion_stat_tests_by_media_type.py` script is also retained.
4. `analyze_monthly_user_mobility.py` and
   `analyze_quarterly_user_mobility.py` produce commenter-mobility summaries.
5. `bootstrap_user_mobility_ci.py` produces the period-resampling interval
   summaries.
6. The `plot_*.py` scripts generate the manuscript figures from these analysis
   outputs. `validate_emotion_model_on_samples.py` produces the emotion
   agreement summaries reported in `validation/`.

`analyze_topic_trends_by_media_type.py` and
`analyze_channel_level_by_media_type.py` generate the quarterly and channel
summaries. `strengthen_core_analyses.py` implements dynamic participation tiers,
the supplementary mobility models, and additional robustness analyses. Its full
entry point requires the non-deposited raw inputs; the figure rebuild command
uses only its plotting function and the released dynamic-tier rates.

## Reproduction scope

The Google Drive package contains the aggregate outputs required to inspect and reproduce the reported descriptive summaries, statistical-test outputs, bootstrap intervals, and supplementary model terms. Platform data may change after collection, so independently recollected raw snapshots need not match the archived aggregates exactly.

## License and privacy

Unless otherwise noted, the code in this repository is released under
the [MIT License](LICENSE).

The non-identifying aggregate data available through Google Drive are
released under the Creative Commons Attribution 4.0 International
License (CC BY 4.0). Reuse must acknowledge this repository and the
associated manuscript.

The public release contains derived aggregate outputs only. It does not
include, license, or convey rights to raw YouTube comment text, public
commenter identifiers, commenter-level linkage data, or other
platform-derived personal data.
