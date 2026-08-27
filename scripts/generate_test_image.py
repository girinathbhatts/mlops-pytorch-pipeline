"""Generate a test image for the /predict endpoint."""

import numpy as np
from PIL import Image


def generate_test_image(output_path: str = "test_image.png"):
    """Generate a random 32x32 RGB image for testing."""
    rng = np.random.default_rng(42)
    img_array = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Test image saved to {output_path}")


if __name__ == "__main__":
    generate_test_image()
