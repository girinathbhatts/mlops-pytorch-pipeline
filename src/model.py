"""Model definitions for CIFAR-10 image classification."""

import torch.nn as nn
from torchvision import models


def get_model(
    architecture: str = "resnet18", num_classes: int = 10
) -> nn.Module:
    """Create and return a model based on the specified architecture.

    Args:
        architecture: Model architecture name ('resnet18' or 'resnet34').
        num_classes: Number of output classes.

    Returns:
        A PyTorch model configured for the specified number of classes.

    Raises:
        ValueError: If an unknown architecture is specified.
    """
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
    elif architecture == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Modify for CIFAR-10 (32x32 images instead of ImageNet 224x224)
    # Replace 7x7 conv with 3x3 conv (less aggressive downsampling)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove max pooling (not needed for small images)
    model.maxpool = nn.Identity()
    # Replace final FC layer for correct number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
