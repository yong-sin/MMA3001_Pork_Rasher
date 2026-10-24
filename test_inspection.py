"""Unit tests for packaging inspection and its input validation."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from inspection import inspect_package


@pytest.fixture
def valid_paths(tmp_path):
    """Create a readable image and a placeholder weights file for mocked tests."""
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (32, 32), color=(120, 80, 60)).save(image_path)
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"placeholder checkpoint for mocked YOLO")
    return image_path, weights_path


def test_no_detections(valid_paths):
    """Return an empty list when the mocked model detects no defects."""
    image_path, weights_path = valid_paths
    fake_model = MagicMock()
    fake_model.predict.return_value = [MagicMock(boxes=[])]

    with patch("inspection.YOLO", return_value=fake_model) as mock_yolo:
        result = inspect_package(image_path, weights_path)

    assert result == []
    mock_yolo.assert_called_once_with(str(weights_path))
    fake_model.predict.assert_called_once_with(
        source=str(image_path), conf=0.201, imgsz=640, verbose=False
    )


def test_one_detection(valid_paths):
    """Return the defect class, confidence and bounding box."""
    image_path, weights_path = valid_paths
    fake_box = MagicMock()
    fake_box.cls.item.return_value = 3
    fake_box.conf.item.return_value = 0.629
    fake_box.xyxy[0].tolist.return_value = [0, 330.8, 325, 539.5]

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]
    fake_result.names = {3: "unsealed"}

    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    with patch("inspection.YOLO", return_value=fake_model):
        result = inspect_package(image_path, weights_path)

    assert result == [{
        "defect": "unsealed",
        "confidence": 0.629,
        "box": [0.0, 330.8, 325.0, 539.5],
    }]


def test_missing_image(valid_paths, tmp_path):
    """Reject a path that does not point to an existing input image."""
    _, weights_path = valid_paths
    with patch("inspection.YOLO") as mock_yolo:
        with pytest.raises(FileNotFoundError, match="Input image not found"):
            inspect_package(tmp_path / "missing.jpg", weights_path)
    mock_yolo.assert_not_called()


def test_missing_weights(valid_paths, tmp_path):
    """Reject a path that does not point to a weights file."""
    image_path, _ = valid_paths
    with patch("inspection.YOLO") as mock_yolo:
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            inspect_package(image_path, tmp_path / "missing.pt")
    mock_yolo.assert_not_called()


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), "high"])
def test_invalid_confidence(valid_paths, confidence):
    """Reject invalid confidence values before loading the model."""
    image_path, weights_path = valid_paths
    with patch("inspection.YOLO") as mock_yolo:
        with pytest.raises(ValueError, match="confidence must"):
            inspect_package(image_path, weights_path, confidence=confidence)
    mock_yolo.assert_not_called()


def test_unreadable_image(valid_paths, tmp_path):
    """Reject a file that exists but does not contain an image."""
    _, weights_path = valid_paths
    broken_image = tmp_path / "corrupted.jpg"
    broken_image.write_bytes(b"This is not JPEG image data")

    with patch("inspection.YOLO") as mock_yolo:
        with pytest.raises(ValueError, match="Input image is unreadable"):
            inspect_package(broken_image, weights_path)
    mock_yolo.assert_not_called()
