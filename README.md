# HAM10000 — Phase 2: Skin Lesion Classification

> **Status:** work in progress. This is the first version of the README. It explains the proposed structure for the second half of the semester. If something is unclear, let me know and we'll clarify it here.

---

## Why a new repository?

In the first half of the semester, all our analysis grew inside a single notebook (`HAM10000_unified_analysis.ipynb`). It ended up with more than 10 sections, historical parts, code from the (now fixed) data leakage, and final results all mixed together. Today it is hard to follow, even for the person who wrote it.

That's why I decided to start a **new repository** for the second half:

- **The previous repository is frozen.** It is the record of what we did and presented in the first half. We won't modify it, so those results stay reproducible exactly as they are.
  Previous repository: `<link>` — final state: commit `18a26db`.
- **This repository starts clean.** There are no legacy files and no mixed-language names, and every file has one clear purpose.

The goal is that anyone on the team (or the professors) can open the repository and understand within a few minutes what each part does and in what order it runs.

---

## The big idea in one sentence

> **Heavy work runs as scripts on the remote machine; light work is analyzed in notebooks.**

| Type of work | Where | How | Examples |
|---|---|---|---|
| **Heavy** (needs a GPU or lots of CPU) | Remote machine | Python scripts inside `tmux` | Preprocessing images, training, generating predictions |
| **Light** (analysis) | Local | `.ipynb` notebooks | Comparing experiments, plots, statistics |

Notebooks **never train or load models**. They only read the result files (`.csv`, `.json`) produced by the scripts, so anyone can open them on a laptop without a GPU.

---

## Repository structure

```
ham10000-phase2/
├── README.md            ← this file
├── requirements.txt     ← libraries with exact versions
│
├── data/
│   ├── README.md        ← how to download HAM10000 and where to put the images
│   └── splits/          ← the official train / val / test split
│
├── configs/             ← one YAML file per experiment
├── src/ham10000/        ← reusable code (the "library")
├── scripts/             ← what runs on the remote machine
├── notebooks/           ← the analyses
├── runs/                ← results of each training run (not pushed to git)
└── docs/                ← protocol and decision log
```

Below is what each folder contains.

### `data/splits/` — the data split

It holds three files (`train.csv`, `val.csv`, `test.csv`) listing the images in each subset. We use **exactly the same split as in the first half** (70/15/15, grouped by lesion, no data leakage), so the new results can be compared with the previous ones.

⚠️ **Never regenerate or modify it.** The images themselves are not pushed to git; see `data/README.md`.

### `configs/` — the recipe for each experiment

Each experiment is described in a small YAML file: which task (binary or multiclass), which preprocessing, which resolution, which model, and so on.

- `base.yaml` holds the default values.
- Each experiment file only lists what changes from the base.

To see how two experiments differ, you just compare two short files.

### `src/ham10000/` — the reusable code

This is where the functions used by the scripts and notebooks live. It is never run directly. Each file covers a single topic:

| File | What it contains |
|---|---|
| `config.py` | Reading and validating configuration files |
| `data.py` | Loading the splits and images, and preparing training batches |
| `preprocessing.py` | The 4 preprocessing pipelines: baseline, DullRazor, Shades of Gray, Contrast-Focused |
| `augmentation.py` | Random transformations applied during training |
| `models.py` | The models (EfficientNet and others) and their binary or multiclass heads |
| `losses.py` | Loss functions (class-weighted, focal loss, etc.) |
| `training.py` | The training loop (phases, early stopping, learning rate) |
| `evaluation.py` | Metrics, thresholds, and lesion-level bootstrap |
| `runs.py` | Creating each run's folder and saving everything needed to reproduce it |

### `scripts/` — what runs on the remote machine

The scripts are **numbered in the order they are run**:

| Script | What it does |
|---|---|
| `00_check_setup.py` | Checks that the images, splits, GPU and library versions are correct |
| `01_cache_preprocessed.py` | Preprocesses the images **only once** and saves them to disk (before, this was recomputed in every training run) |
| `02_train.py` | Trains a model from a file in `configs/` |
| `03_predict.py` | Generates predictions on validation/test and saves them as CSV |
| `launch_grid.sh` | Launches several experiments in a row inside `tmux` |

There is now **a single training script** for both tasks. The task (binary or multiclass) is chosen in the configuration file.



### `notebooks/` — the analyses

| Notebook | Purpose |
|---|---|
| `01_sanity_checks.ipynb` | Check that everything works: the split is correct and we reproduce a first-half result |
| `02_experiments.ipynb` | Compare the new experiments on validation |
| `03_error_analysis.ipynb` | Study errors (e.g. melanomas misclassified as nevi), calibration and Grad-CAM |
| `04_final_test.ipynb` | Final evaluation on the test set. **Run only once, at the very end** |

### `runs/` — the results of each training run

Each training run creates its own folder with everything needed to understand and reproduce it:

```
runs/<task>/<experiment>_<seed>_<date>/
├── config.yaml          ← full configuration used
├── command.txt          ← exact command and code version (git hash)
├── metrics.json         ← final metrics
├── history.csv          ← metrics per epoch
├── predictions_val.csv  ← probabilities per image
├── model.keras          ← the trained model
└── train.log            ← everything printed to the console
```

This folder is not pushed to git because it is large. We copy it from the remote machine to our local machines to analyze it.

### `docs/` — rules and decision log

- **`protocol.md`**: how we will choose the best model. We write it **before** experimenting so the choice is fair. For example: for binary, maximum specificity while keeping sensitivity ≥ 85%; for multiclass, macro-F1.
- **`decisions.md`**: a log of what we tried, why, and what we discarded. It keeps us from repeating work and makes the final report easier to write.

---

## Typical workflow

1. **(Once)** Place the HAM10000 images as described in `data/README.md` and install `requirements.txt`.
2. **(Remote)** Run `00_check_setup.py` to verify that everything is in order.
3. **(Remote)** Run `01_cache_preprocessed.py` for the preprocessing and resolution we are going to use.
4. **(Remote, inside `tmux`)** Run `02_train.py` with the experiment's configuration.
5. **(Remote)** Run `03_predict.py` to save the predictions.
6. **(Local)** Copy the run folder into `runs/` and analyze it in the notebooks.
7. Write down what we learned in `docs/decisions.md`.

---

## Important rules

1. **Do not touch the test set** until the final evaluation (`04_final_test.ipynb`). All decisions are made on validation.
2. **Do not modify `data/splits/`.**
3. **Do not push images, models or the `runs/` folder to git.**
4. **Every new experiment gets its own file in `configs/`.** No changing parameters "by hand" without leaving a trace.
5. **Before comparing with the first half**, we first reproduce a known result (Binary Baseline 224, validation AUC ≈ 0.826) to confirm that the new code behaves like the old one.

---

## Goals for the second half

We start from a binary AUC of ~0.84 and a multiclass macro-F1 of ~0.51 on test. We want to improve with:

- Higher resolutions (300–380 px).
- More fine-tuning and more modern models.
- Better handling of class imbalance.
- Several seeds or cross-validation, so the results are statistically stronger.
- Error analysis, calibration and Grad-CAM.

The details of each experiment will go into `docs/decisions.md` as we progress.

---

*For educational and research use only. This is not a diagnostic tool.*