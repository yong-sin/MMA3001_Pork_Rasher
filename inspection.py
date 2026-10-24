"""Validate an input image and detect visible pork-rasher packaging defects."""

import math
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def inspect_package(image_path, weights_path, confidence=0.201):
    """Run YOLOv8 inference on one image after basic input validation.

    Args:
        image_path: Path to an image readable by Pillow.
        weights_path: Path to a trained YOLO checkpoint (``.pt``).
        confidence: Minimum detection confidence, between 0 and 1.

    Returns:
        list[dict]: A list of detections, each with ``defect`` (class name),
        ``confidence`` (rounded score), and ``box`` (four pixel coordinates
        ``[x1, y1, x2, y2]``). Returns an empty list if none are detected.

    Raises:
        FileNotFoundError: If the image or weights file does not exist.
        ValueError: If confidence is invalid or the image cannot be read.

    Notes:
        File and image checks cannot determine whether a readable image
        actually depicts pork rasher packaging or another unsupported domain.
        Model predictions require separate accuracy validation. YOLO may
        raise library-specific errors for invalid checkpoint contents.
    """
    image_path = Path(image_path)
    weights_path = Path(weights_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Input image is unreadable: {image_path}") from exc

    model = YOLO(str(weights_path))
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=640,
        verbose=False,
    )[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "defect": result.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 3),
            "box": [round(float(x), 1) for x in box.xyxy[0].tolist()],
        })

    return detections
