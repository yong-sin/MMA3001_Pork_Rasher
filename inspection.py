"""Detect pork rasher packaging defects in an image."""

from ultralytics import YOLO


def inspect_package(image_path, weights_path, confidence=0.201):
    """Return detected defect names, confidence scores and bounding boxes."""
    model = YOLO(str(weights_path))
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=640,
        verbose=False
    )[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "defect": result.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 3),
            "box": [round(float(x), 1) for x in box.xyxy[0].tolist()]
        })

    return detections
