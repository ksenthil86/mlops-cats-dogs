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
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()

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
