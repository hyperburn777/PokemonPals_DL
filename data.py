import os
import pickle
from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    AUG_PKL_PATH,
    TEST_DIR,
    IMG_SIZE,
    CHANNELS,
    BATCH,
    VAL_SPLIT,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


def _sorted_subdirs(root: str) -> List[str]:
    """Alphabetically sorted subfolders (Keras uses this for label order)."""
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _normalize_to_unit(x: tf.Tensor) -> tf.Tensor:
    """Cast to float32 and scale to [0,1] if needed."""
    x = tf.cast(x, tf.float32)
    # If max > 1.5, assume [0,255] scale
    x = tf.cond(
        tf.reduce_max(x) > 1.5,
        lambda: x / 255.0,
        lambda: x
    )
    return x


def load_datasets(
    return_class_names: bool = False,
) -> Union[
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]],
]:
    """
    Loads:
      - train/val from augmented pickle (with ~11% of training held out for validation)
      - test from TEST_DIR originals (folder-per-class)

    Returns:
      train_ds, val_ds, test_ds
      (+ class_names if return_class_names=True)
    """
    # -------------------------------
    # 1) TRAIN/VAL from pickle
    # -------------------------------
    with open(AUG_PKL_PATH, "rb") as f:
        X_aug, y_aug = pickle.load(f)

    # Sanity checks
    assert X_aug.ndim == 4, f"Expected X_aug to be 4D (N,H,W,C), got {X_aug.shape}"
    h, w = IMG_SIZE
    assert X_aug.shape[1:3] == (
        h,
        w,
    ), f"X_aug spatial shape {X_aug.shape[1:3]} != {IMG_SIZE}"
    assert (
        X_aug.shape[-1] == CHANNELS
    ), f"X_aug channels {X_aug.shape[-1]} != {CHANNELS}"
    assert y_aug.ndim == 1, f"Expected y_aug to be 1D int labels, got {y_aug.shape}"

    # Normalize to [0,1]
    X_aug = X_aug.astype("float32")
    if X_aug.max() > 1.5:
        X_aug /= 255.0

    # Stratified split: ~11% of training to validation
    if not VAL_SPLIT:
        X_tr = X_aug
        y_tr = y_aug
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_aug,
            y_aug,
            test_size=VAL_SPLIT,
            stratify=y_aug,
            shuffle=True,
            random_state=SEED,
        )

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(buffer_size=min(8192, len(X_tr)), seed=SEED)
        .batch(BATCH)
        .prefetch(AUTOTUNE)
    )

    # -------------------------------
    # 2) TEST from originals on disk
    # -------------------------------
    color_mode = "grayscale" if CHANNELS == 1 else "rgb"
    class_names = _sorted_subdirs(TEST_DIR)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        color_mode=color_mode,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=False,  # deterministic evaluation
        seed=SEED,
    )

    # Create a horizontally flipped version
    flipped_ds = test_ds.map(
        lambda x, y: (tf.image.flip_left_right(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Concatenate the original and flipped datasets
    test_ds_with_flips = test_ds.concatenate(flipped_ds)

    # Prefetch for performance
    test_ds = test_ds_with_flips.prefetch(tf.data.AUTOTUNE)

    # Normalize to [0,1]
    test_ds = test_ds.map(
        lambda x, y: (_normalize_to_unit(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    # Validation set is set to test data unless a VAL_SPLIT > 0 is specified
    val_ds = test_ds
    if VAL_SPLIT:
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(BATCH)
            .prefetch(AUTOTUNE)
        )

    if return_class_names:
        return train_ds, val_ds, test_ds, class_names
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick smoke test / summary when run directly
    train_ds, val_ds, test_ds, class_names = load_datasets(return_class_names=True)
    n_train = sum(len(b[1]) for b in train_ds)
    n_val = sum(len(b[1]) for b in val_ds)
    n_test = sum(len(b[1]) for b in test_ds)
    print(
        f"Classes: {len(class_names)} → {class_names[:5]}{' …' if len(class_names) > 5 else ''}"
    )
    print(f"Samples → train: {n_train}, val: {n_val}, test: {n_test}")
    ### Samples → train: 22379, val: 3197, test: 3197 @ 12% validation split
