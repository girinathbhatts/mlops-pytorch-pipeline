"""FastAPI serving application for CIFAR-10 model inference."""

import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import get_model  # noqa: E402

app = FastAPI(
    title="CIFAR-10 Model Serving",
    description="Serves predictions from a trained CIFAR-10 classifier",
    version="1.0.0",
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Global model variable
model = None
device = torch.device("cpu")


def load_model():
    """Load the trained model from a checkpoint file."""
    global model

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    model_name = os.environ.get("MODEL_NAME", "classifier_v1.pt")
    checkpoint_path = Path(checkpoint_dir) / model_name

    # Fallback to local path for development
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints") / model_name

    if not checkpoint_path.exists():
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}",
            flush=True,
        )
        return

    architecture = os.environ.get("MODEL_ARCH", "resnet18")
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))

    model = get_model(
        architecture=architecture, num_classes=num_classes
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Model loaded from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', 'N/A')}, "
        f"val_acc={checkpoint.get('val_accuracy', 'N/A')})",
        flush=True,
    )


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    load_model()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess an image for model inference.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Preprocessed image tensor with batch dimension.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns 200 if model is loaded and ready for inference.
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "model not loaded",
            },
        )
    return {"status": "healthy"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Run inference on an uploaded image.

    Args:
        image: Uploaded image file.

    Returns:
        JSON with predicted class, confidence, and all class probabilities.
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not loaded"},
        )

    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    probs = probabilities[0].tolist()
    predicted_idx = probabilities[0].argmax().item()

    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "confidence": round(probs[predicted_idx], 4),
        "probabilities": {
            cls: round(prob, 4)
            for cls, prob in zip(CIFAR10_CLASSES, probs)
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
