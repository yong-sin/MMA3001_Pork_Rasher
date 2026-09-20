"""Tests for the package inspection function."""

from unittest.mock import MagicMock, patch
from inspection import inspect_package


def test_no_detections():
    """Return an empty list when the model detects no defects."""
    fake_model = MagicMock()
    fake_model.predict.return_value = [MagicMock(boxes=[])]

    with patch("inspection.YOLO", return_value=fake_model):
        result = inspect_package("example.jpg", "best.pt")

    assert result == []


def test_one_detection():
    """Return the expected fields for a detected defect."""
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
        result = inspect_package("example.jpg", "best.pt")

    assert result == [{
        "defect": "unsealed",
        "confidence": 0.629,
        "box": [0.0, 330.8, 325.0, 539.5]
    }]
