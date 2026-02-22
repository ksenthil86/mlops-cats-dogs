"""
Simple CNN model for binary image classification (Cats vs Dogs).
Architecture: 3 Conv blocks + 2 FC layers + Sigmoid output.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification of 224x224 RGB images.

    Architecture:
        - Conv Block 1: Conv2d(3, 32) -> ReLU -> MaxPool2d
        - Conv Block 2: Conv2d(32, 64) -> ReLU -> MaxPool2d
        - Conv Block 3: Conv2d(64, 128) -> ReLU -> MaxPool2d
        - Flatten
        - FC1: Linear -> ReLU -> Dropout
        - FC2: Linear -> Sigmoid
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers
        # After 3 pools: 224 / 2^3 = 28, so feature map is 28x28x128
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Output tensor of shape (batch, 1) with probabilities.
        """
        x = self.conv_blocks(x)
        x = self.fc_layers[0](x)
        x = x.contiguous()
        x = self.fc_layers[1:](x)
        return x


def get_model() -> SimpleCNN:
    """Create and return a new SimpleCNN instance."""
    return SimpleCNN()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(f"Model architecture:\n{model}")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
