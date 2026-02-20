"""
Unit tests for data preprocessing functions.
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preprocessing import (
    load_and_resize_image,
    normalize_image,
    augment_image,
    split_dataset,
    IMG_SIZE,
)


@pytest.fixture
def sample_image_path():
    """Create a temporary sample image and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_image_array():
    """Create a sample image numpy array."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


class TestLoadAndResizeImage:
    """Tests for load_and_resize_image function."""

    def test_output_shape(self, sample_image_path):
        """Test that output image has correct shape (224, 224, 3)."""
        result = load_and_resize_image(sample_image_path)
        assert result.shape == (IMG_SIZE, IMG_SIZE, 3)

    def test_output_dtype(self, sample_image_path):
        """Test that output is uint8."""
        result = load_and_resize_image(sample_image_path)
        assert result.dtype == np.uint8

    def test_custom_size(self, sample_image_path):
        """Test resizing to a custom size."""
        result = load_and_resize_image(sample_image_path, size=128)
        assert result.shape == (128, 128, 3)

    def test_rgb_conversion(self):
        """Test that grayscale images are converted to RGB."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            gray_img = Image.fromarray(np.random.randint(0, 255, (50, 50), dtype=np.uint8), mode="L")
            gray_img.save(f.name)
            result = load_and_resize_image(f.name)
            assert result.shape == (IMG_SIZE, IMG_SIZE, 3)
        os.unlink(f.name)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_output_range(self, sample_image_array):
        """Test that normalized values are in [0, 1]."""
        result = normalize_image(sample_image_array)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self, sample_image_array):
        """Test that output is float32."""
        result = normalize_image(sample_image_array)
        assert result.dtype == np.float32

    def test_shape_preserved(self, sample_image_array):
        """Test that shape is preserved after normalization."""
        result = normalize_image(sample_image_array)
        assert result.shape == sample_image_array.shape


class TestAugmentImage:
    """Tests for augment_image function."""

    def test_output_shape(self, sample_image_array):
        """Test that augmented image has same shape."""
        result = augment_image(sample_image_array)
        assert result.shape == sample_image_array.shape

    def test_output_dtype(self, sample_image_array):
        """Test that output is uint8."""
        result = augment_image(sample_image_array)
        assert result.dtype == np.uint8

    def test_output_range(self, sample_image_array):
        """Test that pixel values remain in valid range."""
        result = augment_image(sample_image_array)
        assert result.min() >= 0
        assert result.max() <= 255


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_split_sizes(self):
        """Test that splits have correct proportions."""
        files = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(files)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_no_overlap(self):
        """Test that splits don't overlap."""
        files = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(files)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_all_items_present(self):
        """Test that all items are in exactly one split."""
        files = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(files)
        all_split = splits["train"] + splits["val"] + splits["test"]
        assert len(all_split) == len(files)

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        files = [f"img_{i}.jpg" for i in range(100)]
        split1 = split_dataset(files, seed=42)
        split2 = split_dataset(files, seed=42)
        assert split1["train"] == split2["train"]
        assert split1["val"] == split2["val"]
        assert split1["test"] == split2["test"]

    def test_custom_ratios(self):
        """Test custom split ratios."""
        files = [f"img_{i}.jpg" for i in range(100)]
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        splits = split_dataset(files, ratios=ratios)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15
