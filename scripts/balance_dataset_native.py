#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HAM10000 Dataset Preparation and Balancing (native resolution)
==============================================================

This script prepares the HAM10000 dataset for skin lesion
classification at the native image resolution (600 x 450 px,
width x height) while preventing data leakage.

Main steps:
1. Load HAM10000 metadata and images from data/raw/.
2. Load the official lesion-level split from data/splits-v1/
   (or, with --resplit, recompute it by lesion_id, 70/15/15).
3. Verify that no lesion appears in multiple subsets.
4. Apply image preprocessing, keeping the native resolution.
5. Optionally perform data augmentation on training data only.
6. Save training, validation, and test datasets to data/splits-v2/.

Differences with the previous (224 x 224) version:
- Images keep their native 600 x 450 size and aspect ratio.
  With the baseline method, original files are copied byte for
  byte (no resizing and no JPEG re-encoding).
- Augmentation only uses transformations that preserve the
  600 x 450 shape: horizontal flip, vertical flip, and their
  combination (equivalent to a 180-degree rotation). 90 and
  270-degree rotations are not used because they would turn the
  image into 450 x 600.

By default, no augmentation is performed. This allows
controlled comparisons between models trained with and
without online data augmentation.

Usage (from the project root):
    python3 scripts/balance_dataset_native.py

    python3 scripts/balance_dataset_native.py \
        --output-dir data/splits-v2-balanced \
        --balance-training
"""

import argparse
import random
import shutil
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from tqdm import tqdm


# 1. CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits-v1"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits-v2"

IMAGE_FOLDERS = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]

# Split names used in the output folder, and the CSV file
# that holds each one in the official split directory.

SPLIT_FILES = {
    "training": "train.csv",
    "validation": "val.csv",
    "test": "test.csv"
}

# Native HAM10000 resolution as (width, height)

NATIVE_IMAGE_SIZE = (600, 450)

DEFAULT_SEED = 42
DEFAULT_TARGET_COUNT = 1000


# 2. FIND IMAGE PATHS


def get_image_paths(base_dir):

    """
    Find all HAM10000 images in the two original directories.

    Returns:
        Dictionary mapping image_id to image path.
    """

    image_paths = {}

    for folder_name in IMAGE_FOLDERS:

        folder = base_dir / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {folder}"
            )

        for image_path in folder.glob("*.jpg"):

            image_id = image_path.stem

            if image_id in image_paths:
                raise ValueError(
                    f"Duplicate image ID found: {image_id}"
                )

            image_paths[image_id] = image_path

    return image_paths



# 3. LOAD OR COMPUTE THE LESION-LEVEL SPLIT


def check_no_leakage(splits):

    """
    Verify that no lesion or image is shared between subsets.
    """

    for split_a, split_b in [
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test")
    ]:

        for column, label in [
            ("lesion_id", "lesions"),
            ("image_id", "images")
        ]:

            values_a = set(splits[split_a][column])
            values_b = set(splits[split_b][column])

            overlap = values_a.intersection(values_b)

            if overlap:
                raise RuntimeError(
                    f"DATA LEAKAGE DETECTED: "
                    f"{len(overlap)} {label} shared between "
                    f"{split_a} and {split_b}"
                )


def load_split(metadata, split_dir):

    """
    Load the official split from split_dir (train/val/test CSV
    files with image_id, lesion_id and dx columns).

    The split is checked against the raw metadata: every image
    must appear exactly once, and lesion_id and dx must match.
    """

    print(f"\nLoading official split from {split_dir}...")

    metadata_by_id = metadata.set_index("image_id")

    splits = {}

    for split_name, file_name in SPLIT_FILES.items():

        csv_path = split_dir / file_name

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {csv_path}"
            )

        split_ids = pd.read_csv(csv_path)

        unknown_ids = set(split_ids["image_id"]) - set(
            metadata_by_id.index
        )

        if unknown_ids:
            raise ValueError(
                f"{len(unknown_ids)} image IDs in {csv_path} "
                "are not in the metadata."
            )

        split_df = (
            metadata_by_id
            .loc[split_ids["image_id"]]
            .reset_index()
        )

        for column in ["lesion_id", "dx"]:

            if not (
                split_df[column].values
                == split_ids[column].values
            ).all():
                raise ValueError(
                    f"Column '{column}' in {csv_path} does not "
                    "match the raw metadata."
                )

        splits[split_name] = split_df

    total_images = sum(len(df) for df in splits.values())

    if total_images != len(metadata):
        raise ValueError(
            f"The split covers {total_images} images, but the "
            f"metadata has {len(metadata)}."
        )

    check_no_leakage(splits)

    print("Official split loaded successfully.")
    print("No overlapping lesion or image IDs detected.")

    return splits


def split_dataset(metadata, seed=42):

    """
    Split the dataset into:
        70% training
        15% validation
        15% testing

    Splitting is performed by lesion_id instead of image_id.

    This ensures that images belonging to the same lesion
    cannot appear in different subsets.
    """

    from sklearn.model_selection import GroupShuffleSplit

    print("\nSplitting dataset by lesion_id...")

    # First split: training (70%) and remaining (30%)

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )

    train_idx, remaining_idx = next(
        first_split.split(
            metadata,
            groups=metadata["lesion_id"]
        )
    )

    train_df = metadata.iloc[train_idx].copy()

    remaining_df = metadata.iloc[remaining_idx].copy()

    # Second split: validation (15%) and testing (15%)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 1
    )

    val_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["lesion_id"]
        )
    )

    val_df = remaining_df.iloc[val_idx].copy()

    test_df = remaining_df.iloc[test_idx].copy()

    splits = {
        "training": train_df,
        "validation": val_df,
        "test": test_df
    }

    check_no_leakage(splits)

    print("Dataset split completed successfully.")
    print("No overlapping lesion IDs detected.")

    return splits



# 4. IMAGE PREPROCESSING


def load_preprocess_pipeline():

    """
    Import preprocess_pipeline from a preprocessing.py module,
    looked up in src/, scripts/ and the Python path.
    """

    for folder in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
        if str(folder) not in sys.path:
            sys.path.append(str(folder))

    try:
        from preprocessing import preprocess_pipeline
    except ImportError as error:
        raise ImportError(
            "Methods other than 'baseline' need a preprocessing.py "
            "module with a preprocess_pipeline function (in src/ "
            "or scripts/)."
        ) from error

    return preprocess_pipeline


def preprocess_image(
    image_path,
    output_path,
    image_size=NATIVE_IMAGE_SIZE,
    method="baseline"
):

    """
    Load, preprocess and save one image.

    Available methods:
        baseline
        hair_removal
        color_constancy
        contrast_focused

    image_size is (width, height). With the baseline method, if
    the image already has that size, the original file is copied
    unchanged, so no quality is lost to JPEG re-encoding. Other
    sizes are resized with Lanczos resampling.

    Other methods use the preprocessing.py module.
    """

    with Image.open(image_path) as image_file:
        original_size = image_file.size
        image = image_file.convert("RGB")

    if method == "baseline":

        if original_size == tuple(image_size):
            shutil.copyfile(image_path, output_path)
            return

        image = image.resize(
            tuple(image_size),
            Image.Resampling.LANCZOS
        )

    else:

        preprocess_pipeline = load_preprocess_pipeline()

        processed_array = preprocess_pipeline(
            np.asarray(image),
            target_size=tuple(image_size),
            method=method
        )

        image = Image.fromarray(
            np.asarray(processed_array, dtype=np.uint8)
        ).convert("RGB")

    image.save(
        output_path,
        quality=95
    )



# 5. DATA AUGMENTATION


def apply_augmentation(image, rng):

    """
    Apply random transformations to a training image.

    Transformations:
        Random horizontal flip
        Random vertical flip

    Applying both flips is equivalent to a 180-degree rotation,
    so the four possible outcomes are: original, horizontal
    flip, vertical flip, and 180-degree rotation. All of them
    preserve the 600 x 450 shape. 90 and 270-degree rotations
    are not used because they would swap width and height.

    Augmentation is never applied to validation or test data.
    """

    # Random horizontal flip

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    # Random vertical flip

    if rng.random() < 0.5:
        image = ImageOps.flip(image)

    return image



# 6. SAVE ORIGINAL IMAGES


def save_split(
    dataframe,
    split_name,
    output_dir,
    image_size,
    method
):

    """
    Save the original images of one dataset subset.

    Output structure:
        output_dir/split_name/class_name/image_id.jpg
    """

    print(f"\nProcessing {split_name} images...")

    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=split_name
    ):

        class_directory = (
            output_dir / split_name / row.dx
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        preprocess_image(
            row.path,
            class_directory / f"{row.image_id}.jpg",
            image_size=image_size,
            method=method
        )



# 7. BALANCE TRAINING DATA ONLY


def balance_training_dataset(
    train_df,
    output_dir,
    image_size=NATIVE_IMAGE_SIZE,
    target_count=1000,
    seed=42
):

    """
    Generate augmented images for minority classes.

    IMPORTANT:
        Only the training set is modified.

        Validation and test sets remain unchanged.

    Classes with fewer than target_count images receive
    additional augmented images.

    Classes with more than target_count images retain
    all their original images.
    """

    print("\nBalancing training dataset...")

    rng = random.Random(seed)

    for class_name, class_df in train_df.groupby("dx"):

        class_directory = (
            output_dir / "training" / class_name
        )

        original_paths = [
            class_directory / f"{image_id}.jpg"
            for image_id in class_df["image_id"]
        ]

        original_paths = [
            path
            for path in original_paths
            if path.exists()
        ]

        original_count = len(original_paths)

        if original_count == 0:
            raise ValueError(
                f"No training images found for {class_name}"
            )

        augmentation_needed = max(
            0,
            target_count - original_count
        )

        print(
            f"\nClass: {class_name}"
            f"\nOriginal images: {original_count}"
            f"\nAugmented images required: {augmentation_needed}"
        )

        for i in tqdm(
            range(augmentation_needed),
            desc=f"Augmenting {class_name}"
        ):

            source_path = rng.choice(original_paths)

            with Image.open(source_path) as image_file:
                image = image_file.convert("RGB")

            augmented_image = apply_augmentation(
                image,
                rng
            )

            # Flips keep the size, so this only acts as a
            # safeguard if a source image has a different size.

            if augmented_image.size != tuple(image_size):
                augmented_image = augmented_image.resize(
                    tuple(image_size),
                    Image.Resampling.LANCZOS
                )

            output_filename = (
                f"{source_path.stem}_aug_{i:05d}.jpg"
            )

            augmented_image.save(
                class_directory / output_filename,
                quality=95
            )

    print("\nTraining augmentation completed.")



# 8. MAIN FUNCTION


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the HAM10000 dataset at native resolution "
            "(600 x 450) using lesion-level splitting without "
            "data leakage."
        )
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing the original HAM10000 files."
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=(
            "Directory with the official split "
            "(train.csv, val.csv, test.csv)."
        )
    )

    parser.add_argument(
        "--resplit",
        action="store_true",
        help=(
            "Recompute the lesion-level split with "
            "GroupShuffleSplit instead of loading --split-dir."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the prepared dataset will be saved."
    )

    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "hair_removal",
            "color_constancy",
            "contrast_focused"
        ],
        default="baseline",
        help="Image preprocessing method."
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(NATIVE_IMAGE_SIZE),
        help=(
            "Output image width and height "
            "(default: native 600 x 450)."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility."
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

    args = parser.parse_args()

    image_size = tuple(args.img_size)

    # Prevent accidental mixing of different experiments

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        parser.error(
            "Output directory is not empty. "
            "Choose a new directory to avoid mixing experiments."
        )

    if min(image_size) < 1 or args.target_count < 1:
        parser.error(
            "Image size and target count must be positive."
        )

    # Load metadata

    metadata_path = (
        args.base_dir / "HAM10000_metadata.csv"
    )

    print("\nLoading HAM10000 metadata...")

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "image_id",
        "lesion_id",
        "dx"
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError(
            "Metadata contains missing image, lesion, or class IDs."
        )

    if metadata["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs found in metadata."
        )

    # Locate original images

    image_paths = get_image_paths(args.base_dir)

    metadata["path"] = metadata["image_id"].map(
        image_paths
    )

    missing_images = metadata["path"].isna().sum()

    if missing_images > 0:
        raise FileNotFoundError(
            f"{missing_images} images listed in the "
            "metadata were not found."
        )

    print(
        f"Total images: {len(metadata)}"
    )

    print(
        f"Unique lesions: {metadata['lesion_id'].nunique()}"
    )

    print(
        f"Output image size: {image_size[0]} x {image_size[1]}"
    )

    # Split dataset before applying augmentation

    if args.resplit:
        datasets = split_dataset(
            metadata,
            seed=args.seed
        )
    else:
        datasets = load_split(
            metadata,
            args.split_dir
        )

    # Save each dataset subset

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_name, split_df in datasets.items():

        save_split(
            split_df,
            split_name,
            args.output_dir,
            image_size,
            args.method
        )

        # Save metadata for reproducibility

        split_df[
            ["image_id", "lesion_id", "dx"]
        ].to_csv(
            args.output_dir / f"{split_name}_metadata.csv",
            index=False
        )

        print(
            f"{split_name}: "
            f"{len(split_df)} images, "
            f"{split_df['lesion_id'].nunique()} unique lesions"
        )

    # Optional augmentation of training data

    if args.balance_training:

        balance_training_dataset(
            datasets["training"],
            args.output_dir,
            image_size=image_size,
            target_count=args.target_count,
            seed=args.seed
        )

    print("\n")
    print("DATASET PREPARATION COMPLETED")
    print("\n")

    print("No lesion-level data leakage detected.")

    print(
        f"Dataset saved to: "
        f"{args.output_dir.resolve()}"
    )

    print(
        "\nNote: validation and test images "
        "were not augmented."
    )



# 9. EXECUTION


if __name__ == "__main__":
    main()
