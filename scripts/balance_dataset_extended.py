#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended Dataset Preparation: HAM10000 + BCN20000 (splits-v3)
==============================================================

This script builds data/splits-v3 by adding the BCN20000 dataset
(Hospital Clinic de Barcelona, ISIC Archive collection 249) to the
HAM10000 split used in splits-v2, while preventing data leakage.

Main steps:
1. Load the HAM10000 official lesion-level split from data/splits-v1/
   (the same split used in splits-v2, so HAM validation and test
   images are unchanged and results stay comparable).
2. Load the BCN20000 metadata and map its ISIC diagnoses to the
   seven HAM10000 classes (see BCN_CLASS_MAP). Images without a
   diagnosis, or whose diagnosis has no HAM10000 equivalent, are
   excluded and listed in excluded_metadata.csv.
3. Split BCN20000 by lesion_id (GroupShuffleSplit, 70/15/15, seed 42),
   the same technique used to build the HAM10000 split.
4. Merge both datasets subset by subset and verify that no lesion or
   image appears in more than one subset.
5. Copy the original files byte for byte (no resizing and no JPEG
   re-encoding): HAM10000 images are 600 x 450, BCN20000 images are
   1024 x 1024. The training script resizes them on the fly.
6. Optionally augment minority classes of the training set only.
7. Save training, validation and test folders plus metadata CSVs
   (with a "source" column) to data/splits-v3/.

Class mapping (BCN20000 diagnosis -> HAM10000 dx):
    Nevus                                   -> nv
    Melanoma, NOS                           -> mel
    Melanoma metastasis                     -> mel  (--metastasis exclude
                                                     drops these instead)
    Basal cell carcinoma                    -> bcc
    Seborrheic keratosis, Solar lentigo     -> bkl
    Solar or actinic keratosis              -> akiec
    Squamous cell carcinoma, NOS            -> akiec
    Dermatofibroma                          -> df
    Vascular (diagnosis_2, no diagnosis_3)  -> vasc
    Scar                                    -> excluded (no HAM class)
    No diagnosis (original BCN test set)    -> excluded

By default, no augmentation is performed, so the output is directly
comparable with data/splits-v2.

Usage (from the project root):
    python3 scripts/balance_dataset_extended.py

    python3 scripts/balance_dataset_extended.py \
        --output-dir data/splits-v3-balanced \
        --balance-training
"""

import argparse
import random
import shutil
import sys

from pathlib import Path

import pandas as pd

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from balance_dataset_native import (  # noqa: E402
    apply_augmentation,
    check_no_leakage,
    get_image_paths,
    load_split,
    split_dataset,
)


# 1. CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HAM_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_BCN_DIR = PROJECT_ROOT / "data" / "raw" / "bcn20000" / "images"

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits-v1"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits-v3"

SPLIT_NAMES = ["training", "validation", "test"]

# BCN20000 diagnosis_3 -> HAM10000 dx

BCN_CLASS_MAP = {
    "Nevus": "nv",
    "Melanoma, NOS": "mel",
    "Melanoma metastasis": "mel",
    "Basal cell carcinoma": "bcc",
    "Seborrheic keratosis": "bkl",
    "Solar lentigo": "bkl",
    "Solar or actinic keratosis": "akiec",
    "Squamous cell carcinoma, NOS": "akiec",
    "Dermatofibroma": "df",
}

# Vascular lesions have diagnosis_2 but no diagnosis_3

BCN_VASCULAR_DIAGNOSIS_2 = "Benign soft tissue proliferations - Vascular"

OUTPUT_COLUMNS = ["image_id", "lesion_id", "dx", "source", "source_dx"]

DEFAULT_SEED = 42
DEFAULT_TARGET_COUNT = 1000


# 2. LOAD HAM10000 (official split)


def load_ham(ham_dir, split_dir):

    """
    Load HAM10000 metadata and its official lesion-level split.

    Returns:
        Dictionary split_name -> dataframe with OUTPUT_COLUMNS + path.
    """

    print("\nLoading HAM10000...")

    metadata = pd.read_csv(ham_dir / "HAM10000_metadata.csv")

    image_paths = get_image_paths(ham_dir)

    metadata["path"] = metadata["image_id"].map(image_paths)

    if metadata["path"].isna().any():
        raise FileNotFoundError(
            f"{metadata['path'].isna().sum()} HAM10000 images "
            "listed in the metadata were not found."
        )

    splits = load_split(metadata, split_dir)

    for split_name, split_df in splits.items():
        split_df["source"] = "HAM10000"
        split_df["source_dx"] = split_df["dx"]
        splits[split_name] = split_df[OUTPUT_COLUMNS + ["path"]]

    print(
        f"HAM10000: {len(metadata)} images, "
        f"{metadata['lesion_id'].nunique()} lesions"
    )

    return splits


# 3. LOAD AND MAP BCN20000


def load_bcn(bcn_dir, metastasis):

    """
    Load BCN20000 metadata and map diagnoses to HAM10000 classes.

    Returns:
        (labelled dataframe with OUTPUT_COLUMNS + path,
         excluded dataframe with the reason for exclusion)
    """

    print("\nLoading BCN20000...")

    metadata = pd.read_csv(bcn_dir / "metadata.csv")

    if metadata["isic_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in BCN20000 metadata.")

    if metadata["lesion_id"].isna().any():
        raise ValueError("BCN20000 metadata has missing lesion IDs.")

    vascular = (
        metadata["diagnosis_3"].isna()
        & (metadata["diagnosis_2"] == BCN_VASCULAR_DIAGNOSIS_2)
    )

    metadata["source_dx"] = metadata["diagnosis_3"].where(
        ~vascular,
        "Vascular lesion"
    )

    class_map = dict(BCN_CLASS_MAP)
    class_map["Vascular lesion"] = "vasc"

    if metastasis == "exclude":
        del class_map["Melanoma metastasis"]

    metadata["dx"] = metadata["source_dx"].map(class_map)

    metadata["exclusion_reason"] = None

    metadata.loc[
        metadata["source_dx"].isna(),
        "exclusion_reason"
    ] = "no diagnosis"

    metadata.loc[
        metadata["source_dx"].notna() & metadata["dx"].isna(),
        "exclusion_reason"
    ] = "no HAM10000 equivalent"

    metadata = metadata.rename(columns={"isic_id": "image_id"})
    metadata["source"] = "BCN20000"
    metadata["path"] = metadata["image_id"].map(
        lambda image_id: bcn_dir / f"{image_id}.jpg"
    )

    missing = [p for p in metadata["path"] if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} BCN20000 images listed in the "
            f"metadata were not found (e.g. {missing[0]})."
        )

    excluded = metadata[metadata["exclusion_reason"].notna()]
    labelled = metadata[metadata["exclusion_reason"].isna()]

    print(
        f"BCN20000: {len(metadata)} images, "
        f"{metadata['lesion_id'].nunique()} lesions"
    )
    print(
        f"  kept: {len(labelled)} images, "
        f"{labelled['lesion_id'].nunique()} lesions"
    )
    print("  excluded:")
    print(
        excluded
        .groupby(["exclusion_reason", "source_dx"], dropna=False)
        .size()
        .to_string()
    )

    return (
        labelled[OUTPUT_COLUMNS + ["path"]].reset_index(drop=True),
        excluded[
            ["image_id", "lesion_id", "source_dx", "exclusion_reason"]
        ]
    )


# 4. SAVE IMAGES


def save_split(dataframe, split_name, output_dir):

    """
    Copy the original images of one subset, byte for byte.

    Output structure:
        output_dir/split_name/dx/image_id.jpg
    """

    print(f"\nCopying {split_name} images...")

    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=split_name
    ):

        class_directory = output_dir / split_name / row.dx

        class_directory.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(
            row.path,
            class_directory / f"{row.image_id}.jpg"
        )


def balance_training_dataset(train_df, output_dir, target_count, seed):

    """
    Generate augmented images (flips) for minority training classes.

    Flips keep each image's own size, so HAM10000 images stay
    600 x 450 and BCN20000 images stay 1024 x 1024. Validation and
    test sets are never modified.
    """

    print("\nBalancing training dataset...")

    rng = random.Random(seed)

    for class_name, class_df in train_df.groupby("dx"):

        class_directory = output_dir / "training" / class_name

        original_paths = [
            class_directory / f"{image_id}.jpg"
            for image_id in class_df["image_id"]
        ]

        augmentation_needed = max(0, target_count - len(original_paths))

        print(
            f"Class {class_name}: {len(original_paths)} original, "
            f"{augmentation_needed} augmented"
        )

        for i in tqdm(
            range(augmentation_needed),
            desc=f"Augmenting {class_name}"
        ):

            source_path = rng.choice(original_paths)

            with Image.open(source_path) as image_file:
                image = image_file.convert("RGB")

            apply_augmentation(image, rng).save(
                class_directory / f"{source_path.stem}_aug_{i:05d}.jpg",
                quality=95
            )


# 5. SUMMARY


def class_count_table(datasets):

    """
    Images per class, source and subset.
    """

    rows = []

    for split_name, split_df in datasets.items():

        counts = (
            split_df
            .groupby(["source", "dx"])
            .size()
            .rename("images")
            .reset_index()
        )

        counts.insert(0, "split", split_name)

        rows.append(counts)

    return pd.concat(rows, ignore_index=True)


# 6. MAIN FUNCTION


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build splits-v3: HAM10000 (official split) + BCN20000 "
            "(split by lesion_id), without data leakage."
        )
    )

    parser.add_argument(
        "--ham-dir",
        type=Path,
        default=DEFAULT_HAM_DIR,
        help="Directory containing the original HAM10000 files."
    )

    parser.add_argument(
        "--bcn-dir",
        type=Path,
        default=DEFAULT_BCN_DIR,
        help="Directory with BCN20000 images and metadata.csv."
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory with the official HAM10000 split."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the prepared dataset will be saved."
    )

    parser.add_argument(
        "--metastasis",
        choices=["mel", "exclude"],
        default="mel",
        help="How to treat BCN20000 'Melanoma metastasis' images."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the BCN20000 split and augmentation."
    )

    parser.add_argument(
        "--balance-training",
        action="store_true",
        help="Generate augmented images for minority training classes."
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Minimum number of training images per class."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report the split without copying images."
    )

    args = parser.parse_args()

    if (
        not args.dry_run
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        parser.error(
            "Output directory is not empty. "
            "Choose a new directory to avoid mixing experiments."
        )

    ham_splits = load_ham(args.ham_dir, args.split_dir)

    bcn_df, bcn_excluded = load_bcn(args.bcn_dir, args.metastasis)

    # Same technique as the HAM10000 split: GroupShuffleSplit by
    # lesion_id, 70 / 15 / 15

    bcn_splits = split_dataset(bcn_df, seed=args.seed)

    datasets = {
        split_name: pd.concat(
            [ham_splits[split_name], bcn_splits[split_name]],
            ignore_index=True
        )
        for split_name in SPLIT_NAMES
    }

    check_no_leakage(datasets)

    print("\nMerged split: no lesion or image shared between subsets.")

    table = class_count_table(datasets)

    print(
        "\n"
        + table.pivot_table(
            index=["split", "source"],
            columns="dx",
            values="images",
            fill_value=0,
            margins=True,
            margins_name="total",
            aggfunc="sum"
        ).to_string()
    )

    for split_name, split_df in datasets.items():
        print(
            f"{split_name}: {len(split_df)} images, "
            f"{split_df['lesion_id'].nunique()} lesions, "
            f"mel {(split_df['dx'] == 'mel').mean():.1%}"
        )

    if args.dry_run:
        print("\nDry run: no files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in datasets.items():

        save_split(split_df, split_name, args.output_dir)

        split_df[OUTPUT_COLUMNS].to_csv(
            args.output_dir / f"{split_name}_metadata.csv",
            index=False
        )

    bcn_excluded.to_csv(
        args.output_dir / "excluded_metadata.csv",
        index=False
    )

    table.to_csv(
        args.output_dir / "class_counts.csv",
        index=False
    )

    if args.balance_training:
        balance_training_dataset(
            datasets["training"],
            args.output_dir,
            target_count=args.target_count,
            seed=args.seed
        )

    print("\nDATASET PREPARATION COMPLETED")
    print(f"Dataset saved to: {args.output_dir.resolve()}")
    print("Validation and test images were not augmented.")


# 7. EXECUTION


if __name__ == "__main__":
    main()
