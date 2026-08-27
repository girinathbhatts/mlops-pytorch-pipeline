"""Unit tests for model and dataset modules."""

import pytest
import torch

from model import get_model
from dataset import get_transforms


class TestGetModel:
    """Tests for the get_model function."""

    def test_resnet18_creation(self):
        """Test that ResNet-18 model is created successfully."""
        model = get_model(architecture="resnet18", num_classes=10)
        assert model is not None

    def test_resnet18_output_shape(self):
        """Test that ResNet-18 produces correct output shape."""
        model = get_model(architecture="resnet18", num_classes=10)
        model.eval()
        dummy_input = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (4, 10)

    def test_resnet34_creation(self):
        """Test that ResNet-34 model is created successfully."""
        model = get_model(architecture="resnet34", num_classes=10)
        assert model is not None

    def test_custom_num_classes(self):
        """Test model creation with different number of classes."""
        model = get_model(architecture="resnet18", num_classes=100)
        model.eval()
        dummy_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 100)

    def test_unknown_architecture_raises(self):
        """Test that unknown architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model(architecture="unknown_model")

    def test_model_save_and_load(self, tmp_path):
        """Test that model can be saved and loaded correctly."""
        model = get_model(architecture="resnet18", num_classes=10)
        save_path = tmp_path / "test_model.pt"
        torch.save(
            {"model_state_dict": model.state_dict()}, save_path
        )

        loaded_model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(save_path, weights_only=False)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify outputs match
        model.eval()
        loaded_model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            original_out = model(dummy_input)
            loaded_out = loaded_model(dummy_input)
        assert torch.allclose(original_out, loaded_out)


class TestGetTransforms:
    """Tests for the get_transforms function."""

    def test_train_transforms(self):
        """Test that train transforms include augmentation."""
        transform = get_transforms(train=True)
        assert transform is not None
        # Should have 4 transforms: flip, crop, to_tensor, normalize
        assert len(transform.transforms) == 4

    def test_eval_transforms(self):
        """Test that eval transforms exclude augmentation."""
        transform = get_transforms(train=False)
        assert transform is not None
        # Should have 2 transforms: to_tensor, normalize
        assert len(transform.transforms) == 2
