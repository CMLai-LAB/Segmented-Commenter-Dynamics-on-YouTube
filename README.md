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

## Reproduction scope

The Google Drive package contains the aggregate outputs required to inspect and reproduce the reported descriptive summaries, statistical-test outputs, bootstrap intervals, and supplementary model terms. Platform data may change after collection, so independently recollected raw snapshots need not match the archived aggregates exactly.

## License and privacy

Unless otherwise noted, the code in this repository is released under
the MIT License; see `LICENSE`.

The non-identifying aggregate data available through Google Drive are
released under the Creative Commons Attribution 4.0 International
License (CC BY 4.0). Reuse must acknowledge this repository and the
associated manuscript.

The public release contains derived aggregate outputs only. It does not
include, license, or convey rights to raw YouTube comment text, public
commenter identifiers, commenter-level linkage data, or other
platform-derived personal data.