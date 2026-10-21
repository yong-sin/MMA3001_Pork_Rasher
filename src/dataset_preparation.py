"""Prepare Roboflow YOLO datasets for training and evaluation.

Images are grouped by the filename before `.rf.`. When a source appears
in multiple splits, priority is given to test, then valid, then train.
Only JPG images are supported.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import shutil
import tempfile
import zipfile

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg"}
MANIFEST_FILE = "preparation_manifest.json"
METHOD_VERSION = "source-prefix-priority-v1"


def source_id(filename: str) -> str:
    """Get the source ID from a Roboflow filename.

    Args:
        filename: Image filename.

    Returns:
        Filename prefix before `.rf.`, or the full filename if absent.
    """
    return Path(filename).name.split(".rf.", 1)[0]


def image_paths(root: Path, split: str) -> list[Path]:
    """List the JPG images in a dataset split.

    Args:
        root: Dataset folder.
        split: `train`, `valid`, or `test`.

    Returns:
        Sorted image paths.

    Raises:
        ValueError: Invalid split name.
        FileNotFoundError: Image folder is missing.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected {SPLITS}")
    folder = Path(root) / split / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image directory: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def source_ids(root: Path, split: str) -> set[str]:
    """Get the unique source IDs in a split.

    Args:
        root: Dataset folder.
        split: `train`, `valid`, or `test`.

    Returns:
        Set of source IDs.
    """
    return {source_id(p.name) for p in image_paths(root, split)}


def load_class_names(dataset_root: Path) -> list[str]:
    """Read the class names from `data.yaml` in ID order.

    Args:
        dataset_root: Folder containing `data.yaml`.

    Returns:
        List of class names.

    Raises:
        ValueError: Missing or inconsistent class names.
    """
    config = yaml.safe_load((Path(dataset_root) / "data.yaml").read_text())
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list) or not names:
        raise ValueError("data.yaml needs a nonempty 'names' list or ID-keyed mapping")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("The 'nc' value does not match the number of class names")
    return [str(name) for name in names]


def audit_dataset(root: Path, *, require_labels: bool = True) -> dict:
    """Check images, labels, and bounding boxes in each split.

    Empty labels are counted as background images. Class counts are based
    on labelled boxes, not the number of images.

    Args:
        root: Dataset folder containing `data.yaml`.
        require_labels: Whether each JPG needs a matching label file.

    Returns:
        Image, background, and class counts for each split.

    Raises:
        FileNotFoundError: Required files or folders are missing.
        ValueError: Unsupported format or invalid annotation.
    """
    root = Path(root)
    names = load_class_names(root)
    result = {"class_names": names, "splits": {}}
    for split in SPLITS:
        label_dir = root / split / "labels"
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {label_dir}")
        image_dir = root / split / "images"
        unsupported = sorted(p.name for p in image_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in {".jpeg", ".png", ".webp"})
        if unsupported:
            raise ValueError(f"Unsupported image extensions in {split}: {unsupported[:5]}; "
                             "review and adapt IMAGE_SUFFIXES before cleaning")

        counts = Counter()
        background_count = 0
        missing_labels = 0
        images = image_paths(root, split)
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                if require_labels:
                    raise FileNotFoundError(f"Missing matching YOLO label: {label}")
                missing_labels += 1
                background_count += 1
                continue
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            if not lines:
                background_count += 1
            for line_number, line in enumerate(lines, start=1):
                try:
                    values = [float(v) for v in line.split()]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric label at {label}:{line_number}") from exc
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Expected five finite YOLO values at {label}:{line_number}")
                cls, x, y, w, h = values
                if not cls.is_integer() or not 0 <= int(cls) < len(names):
                    raise ValueError(f"Invalid class ID at {label}:{line_number}")
                if not (w > 0 and h > 0 and
                        0 <= x-w/2 <= x+w/2 <= 1 and
                        0 <= y-h/2 <= y+h/2 <= 1):
                    raise ValueError(f"Box extends outside image at {label}:{line_number}")
                counts[int(cls)] += 1
        result["splits"][split] = {
            "images": len(images),
            "background_images": background_count,
            "missing_label_files": missing_labels,
            "boxes_by_class": {name: counts[i] for i, name in enumerate(names)},
        }
    return result


def print_audit(audit: dict) -> None:
    """Print the image and annotation counts.

    Args:
        audit: Result from `audit_dataset()`.
    """
    for split, info in audit["splits"].items():
        print(f"\n{split.upper()}: {info['images']} images; "
              f"{info['background_images']} backgrounds; "
              f"{info['missing_label_files']} missing label files")
        for name, count in info["boxes_by_class"].items():
            print(f"  {name}: {count} boxes")


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _split_fingerprint(root: Path) -> str:
    """Hash the prepared image and label files."""
    sha = hashlib.sha256()
    for split in SPLITS:
        for image in image_paths(root, split):
            label = root / split / "labels" / f"{image.stem}.txt"
            for item in (image, label):
                sha.update(item.relative_to(root).as_posix().encode("utf-8"))
                if item.exists():
                    sha.update(_hash_file(item).encode("ascii"))
                else:
                    sha.update(b"<MISSING_LABEL>")
    return sha.hexdigest()


def _locate_dataset_root(extracted: Path) -> Path:
    """Find the dataset folder containing `data.yaml`."""
    matches = [p.parent for p in extracted.rglob("data.yaml")
               if all((p.parent / split / "images").is_dir() for split in SPLITS)]
    if len(matches) != 1:
        raise ValueError(f"Expected one YOLO root (data.yaml + train/valid/test), "
                         f"found {len(matches)} in archive")
    return matches[0]


def _remove_cross_split_sources(root: Path, *, require_labels: bool) -> dict[str, int]:
    """Remove repeated sources, keeping test before valid before train."""
    blocked = {
        "train": source_ids(root, "valid") | source_ids(root, "test"),
        "valid": source_ids(root, "test"),
    }
    removed = {"train": 0, "valid": 0}
    for split in ("train", "valid"):
        for image in image_paths(root, split):
            if source_id(image.name) in blocked[split]:
                label = root / split / "labels" / f"{image.stem}.txt"
                if require_labels and not label.exists():
                    raise FileNotFoundError(f"Cannot remove image without label: {label}")
                image.unlink()
                if label.exists():
                    label.unlink()
                removed[split] += 1
    return removed


def _reserve_source_for_validation(root: Path, source: str, expected_count: int | None) -> int:
    """Move one source and its labels from train to valid."""
    selected = [p for p in image_paths(root, "train") if source_id(p.name) == source]
    if not selected or (expected_count is not None and len(selected) != expected_count):
        locations = {split: sum(source_id(p.name) == source for p in image_paths(root, split))
                     for split in SPLITS}
        raise ValueError(
            f"Holdout {source!r}: expected {expected_count or 'at least one'} "
            f"train images, found {len(selected)}. Locations: {locations}. "
            "Dataset version or holdout settings may not match the archive."
        )
    for image in selected:
        label = root / "train" / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Holdout is missing its label: {label}")
        shutil.move(str(image), str(root / "valid" / "images" / image.name))
        shutil.move(str(label), str(root / "valid" / "labels" / label.name))
    return len(selected)


def prepare_dataset(
    zip_path: str | Path,
    output_dir: str | Path,
    *,
    holdout_source: str | None = None,
    expected_holdout_images: int | None = None,
    expected_dataset_version: int | None = None,
    require_labels: bool = True,
) -> dict:
    """Extract, clean, and check a Roboflow YOLO dataset.

    Preparation starts from the original ZIP each time. The output folder
    is replaced only after the dataset passes all checks.

    Args:
        zip_path: Original Roboflow ZIP.
        output_dir: Folder for the cleaned dataset.
        holdout_source: Source ID to move from train to valid, if needed.
        expected_holdout_images: Expected number of images to move.
        expected_dataset_version: Roboflow version to check against.
        require_labels: Whether each JPG needs a matching label file.

    Returns:
        Record of the cleaning settings, dataset counts, and hashes.

    Raises:
        FileNotFoundError: Required dataset files are missing.
        ValueError: Dataset version, splits, or labels fail checks.
    """
    zip_path = Path(zip_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    if expected_holdout_images is not None and holdout_source is None:
        raise ValueError("expected_holdout_images requires holdout_source")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_sha = _hash_file(zip_path)

    with tempfile.TemporaryDirectory(prefix="mma3001_prepare_", dir=output_dir.parent) as tmp:
        tmp_root = Path(tmp)
        extracted = tmp_root / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            # Block unsafe paths in the ZIP.
            for item in archive.namelist():
                path = PurePosixPath(item)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive member: {item!r}")
            archive.extractall(extracted)
        original = _locate_dataset_root(extracted)
        config = yaml.safe_load((original / "data.yaml").read_text())
        version = config.get("roboflow", {}).get("version")
        if expected_dataset_version is not None and str(version) != str(expected_dataset_version):
            raise ValueError(f"Expected Roboflow V{expected_dataset_version}, ZIP declares V{version}")

        # Work on a fresh copy of the dataset.
        stage = tmp_root / "prepared"
        shutil.copytree(original, stage)
        removed = _remove_cross_split_sources(stage, require_labels=require_labels)
        heldout = (_reserve_source_for_validation(stage, holdout_source, expected_holdout_images)
                   if holdout_source is not None else 0)
        ids = {split: source_ids(stage, split) for split in SPLITS}
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = ids[a] & ids[b]
            if shared:
                raise ValueError(f"Source leakage between {a} and {b}: {sorted(shared)[:5]}")

        # Set the paths used by YOLO.
        config.update({"path": str(output_dir),
                       "train": "train/images", "val": "valid/images", "test": "test/images"})
        (stage / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        audit = audit_dataset(stage, require_labels=require_labels)
        manifest = {
            "method_version": METHOD_VERSION,
            "dataset_zip": zip_path.name,
            "zip_sha256": source_sha,
            "roboflow_version": version,
            "settings": {
                "source_rule": "prefix before .rf. (JPG only)",
                "split_priority": "test > valid > train",
                "holdout_source": holdout_source,
                "expected_holdout_images": expected_holdout_images,
                "require_labels": require_labels,
            },
            "removed_overlap_images": removed,
            "moved_to_validation": heldout,
            "audit": audit,
            "split_fingerprint_sha256": _split_fingerprint(stage),
        }
        (stage / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if output_dir.exists():
            # Keep the old output if preparation fails.
            shutil.rmtree(output_dir)
        shutil.move(str(stage), str(output_dir))
    return manifest


def assert_preparations_match(
    current: dict | str | Path,
    reference: dict | str | Path,
) -> None:
    """Check whether two dataset preparation records match.

    Args:
        current: Current record or path to its JSON file.
        reference: Reference record or path to its JSON file.

    Raises:
        ValueError: Preparation records differ.

    Matching records confirm the same prepared data, not the training
    history of previously saved model weights.
    """
    def as_manifest(obj):
        return json.loads(Path(obj).read_text()) if isinstance(obj, (str, Path)) else obj

    a, b = as_manifest(current), as_manifest(reference)
    keys = ("method_version", "zip_sha256", "roboflow_version", "settings",
            "removed_overlap_images", "moved_to_validation", "audit",
            "split_fingerprint_sha256")
    mismatches = [key for key in keys if a.get(key) != b.get(key)]
    if mismatches:
        raise ValueError(f"Prepared dataset does not match training manifest: {mismatches}")
