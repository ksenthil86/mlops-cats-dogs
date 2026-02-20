# MLOps Pipeline — Cats vs Dogs Classifier

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform. Built with PyTorch, FastAPI, MLflow, DVC, Docker, Kubernetes, Prometheus, and GitHub Actions.

---

## Project Structure

```
mlops-cats-dogs/
├── data/                          # Dataset (tracked by DVC)
│   ├── raw/                       # Raw images (cats/ and dogs/ subdirs)
│   └── processed/                 # Preprocessed .pkl files
├── src/
│   ├── data_preprocessing.py      # Image loading, resizing, augmentation, splitting
│   ├── model.py                   # Simple CNN architecture
│   └── train.py                   # Training script with MLflow tracking
├── models/
│   └── cats_dogs_cnn.pkl          # Trained model (tracked by DVC)
├── app/
│   ├── main.py                    # FastAPI inference service
│   └── monitoring.py              # Prometheus metrics definitions
├── tests/
│   ├── test_preprocessing.py      # Unit tests for data preprocessing
│   └── test_inference.py          # Unit tests for model and API
├── k8s/
│   ├── deployment.yaml            # Kubernetes Deployment manifest
│   ├── service.yaml               # Kubernetes Service manifest
│   └── prometheus/
│       ├── prometheus-config.yaml       # Prometheus ConfigMap
│       ├── prometheus-deployment.yaml   # Prometheus Deployment
│       └── prometheus-service.yaml      # Prometheus Service (NodePort 30090)
├── scripts/
│   └── smoke_test.sh              # Post-deployment smoke tests
├── .github/workflows/
│   └── ci-cd.yaml                 # GitHub Actions CI/CD pipeline
├── Dockerfile                     # Container image definition
├── requirements.txt               # Python dependencies (version-pinned)
├── dvc.yaml                       # DVC pipeline stages
├── .dvcignore                     # DVC ignore patterns
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

---

## Prerequisites

Before starting, ensure you have the following installed:

| Tool               | Version   | Purpose                          |
|---------------------|-----------|----------------------------------|
| Python              | 3.10+     | Runtime                          |
| Docker Desktop      | Latest    | Containerization + Kubernetes    |
| kubectl             | 1.28+     | Kubernetes CLI                   |
| Git                 | 2.30+     | Source code versioning           |
| DVC                 | 3.30+     | Data versioning                  |
| pip                 | Latest    | Python package manager           |

**Important:** Enable Kubernetes in Docker Desktop:
1. Open Docker Desktop → Settings → Kubernetes
2. Check "Enable Kubernetes"
3. Click "Apply & Restart"
4. Wait until Kubernetes status shows green in the Docker Desktop dashboard

---

## Step-by-Step Setup & Deployment

### 1. Clone the Repository and Install Dependencies

```bash
# Clone the repo
git clone https://github.com/<your-username>/mlops-cats-dogs.git
cd mlops-cats-dogs

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC and Get the Dataset

```bash
# Initialize DVC (first time only)
dvc init

# Download the Cats and Dogs dataset from Kaggle
# Place images in: data/raw/cats/ and data/raw/dogs/
# Each folder should contain the respective .jpg images
mkdir -p data/raw/cats data/raw/dogs

# After placing images, track with DVC
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"
```

**Dataset structure expected:**
```
data/raw/
├── cats/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   └── ...
└── dogs/
    ├── dog.0.jpg
    ├── dog.1.jpg
    └── ...
```

### 3. Preprocess the Data

```bash
python src/data_preprocessing.py
```

This will:
- Load all images from `data/raw/`
- Resize to 224×224 RGB
- Split into 80% train / 10% val / 10% test
- Apply augmentation to training images
- Save as `data/processed/train.pkl`, `val.pkl`, `test.pkl`

### 4. Train the Model with MLflow Tracking

```bash
# Start MLflow tracking server (in a separate terminal)
mlflow ui --port 5000

# Run training
cd src
python train.py
cd ..
```

This will:
- Train the Simple CNN for 10 epochs
- Log all hyperparameters, metrics, and artifacts to MLflow
- Save the trained model to `models/cats_dogs_cnn.pkl`
- Generate `loss_curves.png` and `confusion_matrix.png`

**View MLflow dashboard:** Open http://localhost:5000 in your browser.

Track processed data and model with DVC:
```bash
dvc add data/processed
dvc add models/cats_dogs_cnn.pkl
git add data/processed.dvc models/cats_dogs_cnn.pkl.dvc
git commit -m "Track processed data and trained model with DVC"
```

### 5. Run Unit Tests

```bash
pytest tests/ -v
```

Expected output: all tests in `test_preprocessing.py` and `test_inference.py` should pass.

### 6. Build the Docker Image

```bash
# Build the image
docker build -t cats-dogs-classifier:latest .

# Test locally
docker run -d -p 8000:8000 --name cats-dogs-test cats-dogs-classifier:latest

# Verify health endpoint
curl http://localhost:8000/health

# Test prediction (use any cat/dog image)
curl -X POST http://localhost:8000/predict -F "file=@path/to/test_image.jpg"

# Check metrics
curl http://localhost:8000/metrics

# Stop and remove the test container
docker stop cats-dogs-test && docker rm cats-dogs-test
```

### 7. Push Docker Image to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Tag the image
docker tag cats-dogs-classifier:latest ksenthil86/cats-dogs-classifier:latest

# Push
docker push ksenthil86/cats-dogs-classifier:latest
```

### 8. Deploy to Local Kubernetes Cluster (Docker Desktop)

**⚠️ Note:** The image (6.2GB) may exceed Docker Desktop's default memory allocation. To deploy on Kubernetes:
1. Increase Docker Desktop memory: Settings → Resources → Memory (set to 8+ GB)
2. Reduce replicas to 1 in `k8s/deployment.yaml`
3. Proceed with steps 8a-8c below

For immediate testing without Kubernetes, run the inference service locally:
```bash
# Local testing (no Kubernetes needed)
cd /Users/senthilkumarkuppan/Desktop/mlops-cats-dogs
source venv/bin/activate
PYTHONPATH=app:src uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, test:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/test/test_1.jpg"
```

**8a. Update the image name in the deployment manifest:**

Edit `k8s/deployment.yaml` and replace `<DOCKER_HUB_USERNAME>` with your Docker Hub username:
```yaml
image: <your-dockerhub-username>/cats-dogs-classifier:latest
replicas: 1
```

**8b. Apply Kubernetes manifests:**

```bash
# Verify Kubernetes is running
kubectl cluster-info
kubectl get nodes

# Deploy the inference service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy Prometheus
kubectl apply -f k8s/prometheus/prometheus-config.yaml
kubectl apply -f k8s/prometheus/prometheus-deployment.yaml
kubectl apply -f k8s/prometheus/prometheus-service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
```

**8c. Wait for pods to be ready:**

```bash
kubectl rollout status deployment/cats-dogs-classifier
kubectl rollout status deployment/prometheus
```

### 9. Access the Deployed Service

**Inference Service:**
```bash
# The service is exposed via LoadBalancer on port 80
# On Docker Desktop, this maps to localhost

# Health check
curl http://localhost:80/health

# Prediction
curl -X POST http://localhost:80/predict -F "file=@path/to/test_image.jpg"

# Metrics
curl http://localhost:80/metrics
```

**Prometheus Dashboard:**
```bash
# Prometheus is exposed on NodePort 30090
# Open in browser:
open http://localhost:30090
```

In Prometheus UI, try these queries:
- `inference_requests_total` — Total request count
- `inference_request_latency_seconds_bucket` — Request latency distribution
- `prediction_cat_total` — Number of cat predictions
- `prediction_dog_total` — Number of dog predictions
- `model_loaded` — Whether model is loaded (1 or 0)

### 10. Run Smoke Tests

```bash
# Run against the Kubernetes service
SERVICE_URL=http://localhost:80 bash scripts/smoke_test.sh
```

This verifies:
- Health endpoint returns 200
- Prediction endpoint returns 200 with correct response fields
- Metrics endpoint returns 200

---

## CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline (`.github/workflows/ci-cd.yaml`) runs automatically on every push/PR to `main`.

### Pipeline Stages:

1. **Test** — Install dependencies, run `pytest tests/ -v`
2. **Build & Push** — Build Docker image, push to Docker Hub (only on `main` push)
3. **Deploy** — Update Kubernetes deployment with new image, run smoke tests

### Required GitHub Secrets:

Set these in your GitHub repository → Settings → Secrets and Variables → Actions:

| Secret                 | Description                          |
|------------------------|--------------------------------------|
| `DOCKER_HUB_USERNAME`  | Your Docker Hub username             |
| `DOCKER_HUB_TOKEN`     | Docker Hub access token              |
| `KUBE_CONFIG`          | Base64-encoded kubeconfig file       |

**To get your base64 kubeconfig:**
```bash
cat ~/.kube/config | base64 | tr -d '\n'
```

---

## Useful Kubernetes Commands

```bash
# View logs from the inference service
kubectl logs -l app=cats-dogs-classifier -f

# View logs from Prometheus
kubectl logs -l app=prometheus -f

# Scale replicas
kubectl scale deployment cats-dogs-classifier --replicas=3

# Restart deployment (e.g., after updating image)
kubectl rollout restart deployment/cats-dogs-classifier

# Delete all resources
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/prometheus/prometheus-service.yaml
kubectl delete -f k8s/prometheus/prometheus-deployment.yaml
kubectl delete -f k8s/prometheus/prometheus-config.yaml
```

---

## Tech Stack Summary

| Component             | Tool                            |
|-----------------------|---------------------------------|
| Language              | Python 3.10                     |
| ML Framework          | PyTorch                         |
| Model                 | Simple CNN (3 conv + 2 FC)      |
| Model Format          | .pkl (pickle)                   |
| API Framework         | FastAPI                         |
| Experiment Tracking   | MLflow                          |
| Data Versioning       | DVC                             |
| Code Versioning       | Git                             |
| Containerization      | Docker                          |
| Orchestration         | Kubernetes (Docker Desktop)     |
| CI/CD                 | GitHub Actions                  |
| Monitoring            | Prometheus                      |
| Testing               | pytest                          |

---

## API Endpoints

| Method | Endpoint   | Description                                      |
|--------|-----------|--------------------------------------------------|
| GET    | `/health`  | Returns service health status and model state    |
| POST   | `/predict` | Accepts image file, returns label + probability  |
| GET    | `/metrics` | Prometheus-formatted metrics                     |

### Example `/predict` Response:
```json
{
  "label": "cat",
  "probability": 0.2314,
  "confidence": 0.7686,
  "filename": "test_cat.jpg",
  "timestamp": "2025-01-15T10:30:00.000000"
}
```

### Example `/health` Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-15T10:30:00.000000"
}
```
