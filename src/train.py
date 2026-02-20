"""
Training script for Cats vs Dogs CNN classifier.
Integrates MLflow for experiment tracking: logs parameters, metrics, and artifacts.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import get_model


# ----- Hyperparameters -----
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_SAVE_PATH = "../models/cats_dogs_cnn.pkl"
PROCESSED_DATA_DIR = "../data/processed"


def load_split(split_name: str, data_dir: str = PROCESSED_DATA_DIR):
    """Load a preprocessed data split from pickle."""
    filepath = os.path.join(data_dir, f"{split_name}.pkl")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    # images: (N, 224, 224, 3) -> (N, 3, 224, 224) for PyTorch
    images = data["images"].transpose(0, 3, 1, 2)
    labels = data["labels"]
    return torch.FloatTensor(images), torch.FloatTensor(labels)


def create_dataloader(images, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Create a PyTorch DataLoader from tensors."""
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return loss, accuracy, predictions, and true labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            preds = (outputs >= 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(batch_labels.cpu().numpy().flatten().tolist())

    n_batches = max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return total_loss / n_batches, accuracy, all_preds, all_labels


def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png"):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def train():
    """Main training function with MLflow tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set MLflow experiment
    mlflow.set_experiment("cats-vs-dogs-classification")

    with mlflow.start_run(run_name="simple-cnn-baseline"):
        # Log hyperparameters
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "BCELoss")
        mlflow.log_param("image_size", 224)

        # Load data
        print("Loading data...")
        train_images, train_labels = load_split("train")
        val_images, val_labels = load_split("val")
        test_images, test_labels = load_split("test")

        mlflow.log_param("train_size", len(train_labels))
        mlflow.log_param("val_size", len(val_labels))
        mlflow.log_param("test_size", len(test_labels))

        train_loader = create_dataloader(train_images, train_labels)
        val_loader = create_dataloader(val_images, val_labels, shuffle=False)
        test_loader = create_dataloader(test_images, test_labels, shuffle=False)

        # Initialize model, loss, optimizer
        model = get_model().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        train_losses = []
        val_losses = []

        print("Starting training...")
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch}/{EPOCHS} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.4f}"
            )

        # Final evaluation on test set
        test_loss, test_acc, test_preds, test_labels_list = evaluate(
            model, test_loader, criterion, device
        )
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"\nTest Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

        # Plot and log artifacts
        loss_curve_path = plot_loss_curves(train_losses, val_losses)
        mlflow.log_artifact(loss_curve_path)
        print(f"Loss curves saved to {loss_curve_path}")

        cm_path = plot_confusion_matrix(test_labels_list, test_preds)
        mlflow.log_artifact(cm_path)
        print(f"Confusion matrix saved to {cm_path}")

        # Save model as .pkl
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        model_cpu = model.cpu()
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(model_cpu.state_dict(), f)
        mlflow.log_artifact(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

        # Also log model via MLflow
        mlflow.pytorch.log_model(model_cpu, "model")

        print("\nTraining complete! All artifacts logged to MLflow.")


if __name__ == "__main__":
    train()
