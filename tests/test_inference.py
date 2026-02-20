"""
Unit tests for model inference and API endpoints.
"""

import os
import sys
import io
import pickle
import numpy as np
import pytest
import torch
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from model import SimpleCNN, get_model, count_parameters


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    def test_model_output_shape(self):
        """Test that model outputs correct shape."""
        model = get_model()
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 1)

    def test_model_output_range(self):
        """Test that model output is in [0, 1] (sigmoid)."""
        model = get_model()
        model.eval()
        dummy_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    def test_model_batch_processing(self):
        """Test that model handles different batch sizes."""
        model = get_model()
        model.eval()
        for batch_size in [1, 2, 8]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 1)

    def test_model_has_parameters(self):
        """Test that model has trainable parameters."""
        model = get_model()
        n_params = count_parameters(model)
        assert n_params > 0

    def test_model_serialization(self, tmp_path):
        """Test that model can be saved and loaded as .pkl."""
        model = get_model()
        model.eval()

        # Save
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model.state_dict(), f)

        # Load
        new_model = SimpleCNN()
        with open(model_path, "rb") as f:
            state_dict = pickle.load(f)
        new_model.load_state_dict(state_dict)
        new_model.eval()

        # Verify same output
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = new_model(dummy_input)
        assert torch.allclose(out1, out2)


class TestPreprocessImage:
    """Tests for image preprocessing in the inference pipeline."""

    def test_preprocess_produces_correct_shape(self):
        """Test preprocessing an image to tensor."""
        from main import preprocess_image

        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        tensor = preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_normalizes_values(self):
        """Test that preprocessed values are in [0, 1]."""
        from main import preprocess_image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        tensor = preprocess_image(image_bytes)
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client with a dummy model."""
        from fastapi.testclient import TestClient

        # Create and save a dummy model
        model = get_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model.state_dict(), f)

        os.environ["MODEL_PATH"] = str(model_path)

        # Re-import to pick up new MODEL_PATH
        import importlib
        import main as main_module

        importlib.reload(main_module)

        with TestClient(main_module.app) as client:
            yield client

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_has_timestamp(self, client):
        """Test that health response includes timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
