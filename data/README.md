# `data/` — HAM10000 dataset and official split

This folder holds everything the project needs to train and evaluate models:

- **`raw/`**: the original, untouched HAM10000 dataset (images and metadata dataframe). It is the starting point for every future training run and for any new preprocessing. Keep it here even when it is not in use, in case it is needed again.
- **`splits/`**: the official train / validation / test split, shared by the whole team.

```
data/
├── README.md                     ← this file
├── raw/                          ← NOT in git (2.6 GB)
│   ├── HAM10000_metadata.csv     ← raw metadata dataframe (one row per image)
│   ├── HAM10000_images_part_1/   ← 5,000 original images
│   └── HAM10000_images_part_2/   ← 5,015 original images
└── splits/                       ← in git
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## `raw/` — the original dataset

The files are exactly as distributed with HAM10000: **10,015 dermoscopic images** (`.jpg`, 600 × 450 px) of **7,470 lesions**, plus their metadata. Nothing in this folder has been resized, preprocessed or augmented.

> ⚠️ **Treat `raw/` as read-only.** Never overwrite, resize or delete images here. Preprocessed or resized versions are generated from these files by `scripts/01_cache_preprocessed.py` and stored elsewhere, so that any resolution (224, 300, 380 px…) or pipeline can be rebuilt at any time.

The images are split into two folders only because the dataset is distributed that way. Each image appears in exactly one of them, so code should look for an image in both folders.

### `HAM10000_metadata.csv`

The raw metadata dataframe, with one row per image (10,015 rows):

| Column | Description |
|---|---|
| `lesion_id` | Lesion identifier (`HAM_xxxxxxx`). One lesion can have several images. |
| `image_id` | Image identifier (`ISIC_xxxxxxx`); the file is `<image_id>.jpg` |
| `dx` | Diagnosis (the label; see table below) |
| `dx_type` | How the diagnosis was confirmed (`histo`, `follow_up`, `consensus`, `confocal`) |
| `age` | Patient age (may be missing) |
| `sex` | Patient sex |
| `localization` | Body site of the lesion |

Diagnosis classes (`dx`):

| Code | Diagnosis |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

### If `raw/` is missing

`raw/` is not pushed to git because of its size. To set it up on a new machine (for example, the remote machine):

1. Download HAM10000 from one of these sources:
   - Harvard Dataverse (official): <https://doi.org/10.7910/DVN/DBW86T>
   - Kaggle mirror: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>
2. Place the two image folders and the metadata file in `data/raw/` using exactly the names shown in the tree above.
3. Run `scripts/00_check_setup.py`, which checks that the images and the split are in place.

Alternatively, copy `data/raw/` directly from a machine that already has it (e.g. with `rsync`).

---

## `splits/` — the official split

These three files define which images belong to each subset. They are the **same split used in the first half of the semester**, copied unchanged from the previous repository (`Datasets_safe/{training,validation,test}_metadata.csv`). Results from both halves can therefore be compared directly.

Each file has three columns: `image_id`, `lesion_id`, `dx`.

| File | Images | Lesions | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train.csv` | 7,002 (69.9%) | 5,229 | 207 | 381 | 741 | 89 | 770 | 4,718 | 96 |
| `val.csv` | 1,503 (15.0%) | 1,120 | 67 | 74 | 172 | 18 | 162 | 988 | 22 |
| `test.csv` | 1,510 (15.1%) | 1,121 | 53 | 59 | 186 | 8 | 181 | 999 | 24 |
| **Total** | **10,015** | **7,470** | 327 | 514 | 1,099 | 115 | 1,113 | 6,705 | 142 |

How it was built and verified:

- It is grouped **by lesion** (`lesion_id`, using `GroupShuffleSplit` with seed 42 in `balance_dataset_new.py` of the previous repository). All images of a lesion are in the same subset, so there is **no data leakage** between train, validation and test.
- It was checked that no lesion or image appears in more than one subset, that together the three files cover all 10,015 images, and that `lesion_id` and `dx` match `raw/HAM10000_metadata.csv`.
- The split is **not class-balanced**: `nv` makes up about two thirds (66–67%) of each subset. Imbalance is handled during training (class weights, losses, sampling), never by changing these files.

> ⚠️ **Never regenerate or modify `splits/`.** All results depend on it.
> **Do not use `test.csv`** until the final evaluation (`notebooks/04_final_test.ipynb`). All decisions are made on validation.

---

## Citation and license

HAM10000 is released under the **CC BY-NC-SA 4.0** license (non-commercial use). If you use it, cite:

> Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018). <https://doi.org/10.1038/sdata.2018.161>
