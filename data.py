import os
import tensorflow as tf
import matplotlib.pyplot as plt

from config import TEST_DIR, IMG_SIZE, CHANNELS, BATCH, VAL_SPLIT, SEED, TRAIN_SIZE

AUTOTUNE = tf.data.AUTOTUNE

rot_layer = tf.keras.layers.RandomRotation(
    factor=0.1,
    fill_mode="constant",
    fill_value=1.0,
)  # ±10% of 2π (~±36°)
trans_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.1,
    width_factor=0.1,
    fill_mode="constant",
    fill_value=1.0,
)  # up to 10% shift
zoom_layer = tf.keras.layers.RandomZoom(
    (-0.1, 0.1),
    fill_mode="constant",
    fill_value=1.0,
)


def _augment_image(image, max_attempts=5):
    """
    Applies random augmentations to a single image tensor without using tf-addons.

    Args:
        image (tf.Tensor): input image
        max_attempts (int): maximum recursive attempts to avoid infinite recursion

    Returns:
        tf.Tensor: augmented image
    """
    image = tf.ensure_shape(image, [None, None, None, CHANNELS])
    original = image

    image = rot_layer(image)
    image = trans_layer(image)
    image = zoom_layer(image)

    # Check if the augmented image is identical to the original
    if tf.reduce_max(tf.abs(image - original)) < 1e-6:
        if max_attempts > 0:
            return _augment_image(original, max_attempts=max_attempts - 1)
        else:
            return image

    return image


def create_augmented_dataset(test_ds, augmentations_per_image=1):
    # Repeat dataset for each augmentation
    ds_repeated = test_ds.repeat(augmentations_per_image)

    # Apply augmentation and normalization
    ds_augmented = ds_repeated.map(
        lambda x, y: (1.0 - _normalize_to_unit(_augment_image(x)), y),
        num_parallel_calls=AUTOTUNE,
    )

    # Prefetch for pipeline performance
    ds_augmented = ds_augmented.prefetch(AUTOTUNE)

    return ds_augmented


def split_dataset(dataset, val_fraction=0.2):
    # Count total elements
    total_count = dataset.cardinality().numpy()
    if total_count == tf.data.INFINITE_CARDINALITY:
        raise ValueError(
            "Dataset has infinite cardinality; please batch/limit it first."
        )

    val_size = int(total_count * val_fraction)

    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)

    return train_ds, val_ds


def _sorted_subdirs(root):
    """Alphabetically sorted subfolders (Keras uses this for label order)."""
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _normalize_to_unit(x):
    """Cast to float32 and scale to [0,1] if needed."""
    x = tf.cast(x, tf.float32)
    # If max > 1.5, assume [0,255] scale
    x = tf.cond(tf.reduce_max(x) > 1.5, lambda: x / 255.0, lambda: x)
    return x


def load_datasets(
    return_class_names=False,
):
    """
    Loads:
      - test from TEST_DIR originals (folder-per-class)
      - train/val from augmenting test

    Returns:
      train_ds, val_ds, test_ds
      (+ class_names if return_class_names=True)
    """
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

    flipped_ds = test_ds.map(
        lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE
    )

    test_ds_aug = test_ds.concatenate(flipped_ds)

    test_ds = test_ds_aug.map(
        lambda x, y: (1.0 - _normalize_to_unit(x), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    aug_ds = create_augmented_dataset(test_ds, augmentations_per_image=TRAIN_SIZE)
    aug_ds = aug_ds.shuffle(buffer_size=len(aug_ds))

    val_ds = test_ds
    if VAL_SPLIT:
        train_ds, val_ds = split_dataset(aug_ds, val_fraction=VAL_SPLIT)
    else:
        train_ds, val_ds = aug_ds, test_ds

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

    for images, labels in train_ds.take(1):
        # Plot first 9 images
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(
                images[i].numpy().squeeze(),
                cmap="gray" if images.shape[-1] == 1 else None,
            )
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
        plt.savefig(f"train_samples.png")
        plt.close()

    for images, labels in test_ds.take(1):
        # Plot first 9 images
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(
                images[i].numpy().squeeze(),
                cmap="gray" if images.shape[-1] == 1 else None,
            )
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
        plt.savefig(f"test_samples.png")
        plt.close()
    ### Samples → train: 22379, val: 3197, test: 3197 @ 12% validation split
