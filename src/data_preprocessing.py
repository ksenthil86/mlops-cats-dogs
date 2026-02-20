"""
Data preprocessing module for Cats vs Dogs classification.
Handles image loading, resizing, augmentation, and train/val/test splitting.
"""

import os
import pickle
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path


# Constants
IMG_SIZE = 224
CHANNELS = 3
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42


def load_and_resize_image(image_path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    Load an image from disk and resize it to (size x size) RGB.

    Args:
        image_path: Path to the image file.
        size: Target size (width and height).

    Returns:
        Numpy array of shape (size, size, 3) with values in [0, 255].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.

    Args:
        image: Numpy array with pixel values in [0, 255].

    Returns:
        Numpy array with pixel values in [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random data augmentation to an image.
    Augmentations: horizontal flip, brightness adjustment, slight rotation.

    Args:
        image: Numpy array of shape (H, W, 3) with values in [0, 255].

    Returns:
        Augmented image as numpy array.
    """
    img = Image.fromarray(image)

    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)

    # Random slight rotation
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=(0, 0, 0))

    return np.array(img, dtype=np.uint8)


def split_dataset(file_list: list, ratios: dict = None, seed: int = RANDOM_SEED) -> dict:
    """
    Split a list of file paths into train/val/test sets.

    Args:
        file_list: List of file paths.
        ratios: Dict with 'train', 'val', 'test' ratios (must sum to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys containing file path lists.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    shuffled = file_list.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def preprocess_dataset(raw_dir: str, processed_dir: str, augment_train: bool = True):
    """
    Full preprocessing pipeline:
    1. Load images from raw_dir (expects subdirs 'cats' and 'dogs')
    2. Resize to 224x224 RGB
    3. Split into train/val/test
    4. Apply augmentation to training set
    5. Save processed data as pickle files

    Args:
        raw_dir: Path to raw dataset directory.
        processed_dir: Path to save processed data.
        augment_train: Whether to apply augmentation to training data.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    # Collect all image file paths and labels
    image_files = []
    labels = []

    for label, class_name in enumerate(["cats", "dogs"]):
        class_dir = raw_path / class_name
        if not class_dir.exists():
            # Also try singular names
            class_dir = raw_path / class_name.rstrip("s")
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} not found, skipping.")
            continue

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_files.append(str(img_file))
                labels.append(label)

    # Create paired list and split
    paired = list(zip(image_files, labels))
    random.seed(RANDOM_SEED)
    random.shuffle(paired)

    n = len(paired)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": paired[:train_end],
        "val": paired[train_end:val_end],
        "test": paired[val_end:],
    }

    for split_name, split_data in splits.items():
        images = []
        split_labels = []

        for img_path, label in split_data:
            try:
                img = load_and_resize_image(img_path)

                if split_name == "train" and augment_train:
                    img = augment_image(img)

                img = normalize_image(img)
                images.append(img)
                split_labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        data = {
            "images": np.array(images),
            "labels": np.array(split_labels),
        }

        output_file = proc_path / f"{split_name}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

        print(f"{split_name}: {len(images)} images saved to {output_file}")


if __name__ == "__main__":
    preprocess_dataset(
        raw_dir="data/raw",
        processed_dir="data/processed",
        augment_train=True,
    )
