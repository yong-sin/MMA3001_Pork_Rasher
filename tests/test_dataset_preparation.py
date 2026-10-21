"""Tests for dataset cleaning using small example YOLO datasets."""

from pathlib import Path
import sys
import zipfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_preparation import (  # noqa: E402
    assert_preparations_match,
    audit_dataset,
    prepare_dataset,
    source_id,
)


def make_dataset_zip(tmp_path: Path, version: int = 4) -> Path:
    """Create a small dataset ZIP for testing."""
    files = {
        "data.yaml": yaml.safe_dump({
            "nc": 2, "names": ["packaging-error", "wrinkle"],
            "roboflow": {"version": version},
        }),
    }
    filenames = {
        "train": ["a.rf.1.jpg", "a.rf.2.jpg", "b.rf.1.jpg", "c.rf.1.jpg",
                  "wrinkle_source.rf.1.jpg", "wrinkle_source.rf.2.jpg",
                  "wrinkle_source.rf.3.jpg", "unique.rf.1.jpg"],
        "valid": ["b.rf.2.jpg", "c.rf.2.jpg"],
        "test": ["c.rf.3.jpg"],
    }
    for split, images in filenames.items():
        for name in images:
            files[f"{split}/images/{name}"] = "not a real jpeg, content only used for checksum"
            cls = 1 if name.startswith("wrinkle_source") else 0
            files[f"{split}/labels/{Path(name).stem}.txt"] = f"{cls} 0.5 0.5 0.2 0.3\n"
    zip_path = tmp_path / f"v{version}.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, content in sorted(files.items()):
            z.writestr(name, content)
    return zip_path


def test_source_ids():
    assert source_id("frame_with_red_8644_jpg.rf.hash.jpg") == "frame_with_red_8644_jpg"
    assert source_id("independent.jpg") == "independent.jpg"


def test_prepare_v4_holdout_and_reproducibility(tmp_path):
    zip_path = make_dataset_zip(tmp_path)
    out = tmp_path / "prepared"
    args = dict(holdout_source="wrinkle_source", expected_holdout_images=3,
                expected_dataset_version=4)
    report1 = prepare_dataset(zip_path, out, **args)
    assert report1["removed_overlap_images"] == {"train": 2, "valid": 1}
    assert report1["moved_to_validation"] == 3
    assert {k: x["images"] for k, x in report1["audit"]["splits"].items()} == {
        "train": 3, "valid": 4, "test": 1,
    }
    assert report1["audit"]["splits"]["test"]["boxes_by_class"]["wrinkle"] == 0
    assert yaml.safe_load((out / "data.yaml").read_text())["path"] == str(out)
    manifest = out / "preparation_manifest.json"
    saved = manifest.read_text()
    report2 = prepare_dataset(zip_path, out, **args)
    assert_preparations_match(report2, report1)
    assert saved == manifest.read_text()


def test_v1_no_extra_holdout(tmp_path):
    archive = make_dataset_zip(tmp_path, version=1)
    report = prepare_dataset(archive, tmp_path / "v1_clean", expected_dataset_version=1)
    assert report["moved_to_validation"] == 0
    assert report["audit"]["splits"]["train"]["boxes_by_class"]["wrinkle"] == 3
    assert audit_dataset(tmp_path / "v1_clean") == report["audit"]


def test_holdout_mismatch_does_not_destroy_prior_output(tmp_path):
    archive = make_dataset_zip(tmp_path)
    out = tmp_path / "prepared"
    before = prepare_dataset(archive, out, holdout_source="wrinkle_source", expected_holdout_images=3)
    with pytest.raises(ValueError, match="Locations"):
        prepare_dataset(archive, out, holdout_source="wrinkle_source", expected_holdout_images=4)
    assert_preparations_match(out / "preparation_manifest.json", before)


def test_provenance_mismatch_detected(tmp_path):
    archive = make_dataset_zip(tmp_path)
    a = prepare_dataset(archive, tmp_path / "a")
    b = prepare_dataset(archive, tmp_path / "b", holdout_source="wrinkle_source",
                        expected_holdout_images=3)
    with pytest.raises(ValueError, match="does not match"):
        assert_preparations_match(a, b)


def test_refuse_wrong_dataset_version(tmp_path):
    archive = make_dataset_zip(tmp_path, version=1)
    with pytest.raises(ValueError, match="Expected Roboflow V4"):
        prepare_dataset(archive, tmp_path / "out", expected_dataset_version=4)
