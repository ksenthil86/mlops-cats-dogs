"""
FastAPI inference service for Cats vs Dogs classification.
Endpoints:
  - GET  /health   : Health check
  - POST /predict  : Image classification
  - GET  /metrics  : Prometheus metrics
"""

import io
import os
import sys
import time
import pickle
import logging
from datetime import datetime

import numpy as np
import torch
from PIL import Image

def _configure_torch_runtime() -> None:
    """Configure torch runtime once per process, safe for module reloads."""
    if os.environ.get("TORCH_RUNTIME_CONFIGURED") == "1":
        return

    torch.set_num_threads(1)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    os.environ["TORCH_RUNTIME_CONFIGURED"] = "1"


_configure_torch_runtime()

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SimpleCNN

from monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_CAT_COUNT,
    PREDICTION_DOG_COUNT,
    MODEL_LOADED,
    get_metrics,
    get_content_type,
)

# ----- Logging Setup -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference-service")

# ----- App Setup -----
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification API for a pet adoption platform",
    version="1.0.0",
)

# ----- Model Loading -----
MODEL_PATH = os.environ.get("MODEL_PATH", "models/cats_dogs_cnn.pkl")
model = None
device = torch.device("cpu")


def load_model():
    """Load the trained CNN model from pickle file."""
    global model
    try:
        net = SimpleCNN()
        with open(MODEL_PATH, "rb") as f:
            state_dict = pickle.load(f)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        model = net
        MODEL_LOADED.set(1)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    load_model()
    logger.info("Inference service started")


# ----- Helper Functions -----
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess an uploaded image for model inference.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Tensor of shape (1, 3, 224, 224).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    img_tensor = torch.FloatTensor(img_array.transpose(2, 0, 1))
    # Add batch dimension
    return img_tensor.unsqueeze(0)


def infer_probability_numpy_fallback(input_tensor: torch.Tensor) -> float:
    """Fallback inference path that avoids torch FC matmul primitives."""
    with torch.no_grad():
        conv_out = model.conv_blocks(input_tensor)

    features = conv_out.reshape(conv_out.shape[0], -1).detach().cpu().numpy().astype(np.float32)

    fc1 = model.fc_layers[1]
    fc2 = model.fc_layers[4]

    weight1 = fc1.weight.detach().cpu().numpy().astype(np.float32).T
    bias1 = fc1.bias.detach().cpu().numpy().astype(np.float32)
    weight2 = fc2.weight.detach().cpu().numpy().astype(np.float32).T
    bias2 = fc2.bias.detach().cpu().numpy().astype(np.float32)

    hidden = np.maximum(features @ weight1 + bias1, 0.0)
    logits = hidden @ weight2 + bias2
    logits = np.clip(logits, -80.0, 80.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return float(probabilities[0, 0])


# ----- Endpoints -----
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    start = time.time()
    status = "healthy" if model is not None else "unhealthy"
    status_code = 200 if model is not None else 503

    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code=status_code).inc()
    REQUEST_LATENCY.labels(endpoint="/health").observe(time.time() - start)

    logger.info(f"Health check: {status}")
    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "model_loaded": model is not None,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.
    Accepts an image file and returns class label and probability.
    """
    start = time.time()

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(device=device, dtype=torch.float32).contiguous()

        # Run inference
        with torch.no_grad():
            try:
                output = model(input_tensor)
                probability = output.item()
            except RuntimeError as runtime_error:
                if "primitive descriptor" not in str(runtime_error).lower():
                    raise
                logger.warning("Falling back to NumPy FC inference due to backend primitive error")
                probability = infer_probability_numpy_fallback(input_tensor)

        # Determine label
        label = "dog" if probability >= 0.5 else "cat"
        confidence = probability if label == "dog" else 1 - probability

        # Update metrics
        if label == "cat":
            PREDICTION_CAT_COUNT.inc()
        else:
            PREDICTION_DOG_COUNT.inc()

        latency = time.time() - start
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code=200).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        # Log request (no sensitive data)
        logger.info(
            f"Prediction: label={label}, confidence={confidence:.4f}, "
            f"latency={latency:.4f}s, filename={file.filename}"
        )

        return {
            "label": label,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "filename": file.filename,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        latency = time.time() - start
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code=500).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "timestamp": datetime.utcnow().isoformat()},
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type=get_content_type())
