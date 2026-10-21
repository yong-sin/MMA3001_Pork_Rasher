# Dataset V2 — Model Evaluation Results

## Dataset

Roboflow Dataset V2 has four classes: loose-meat, packaging-error, twisted-meat and unsealed.
After removing overlapping source identifiers, the dataset contains 1,935 training, 119 validation and 80 test images. No additional wrinkle holdout is applied.

## YOLOv8n Baseline — Test Set

The saved `MMA3001_v2_validation.ipynb` output confirms the following results on 80 test images with 79 labelled instances.

| Metric | Result |
|---|---:|
| Precision | 77.52% |
| Recall | 43.92% |
| mAP50 | 49.85% |
| mAP50–95 | 24.34% |

Twisted-meat recall was 0 out of four labelled instances. This limits conclusions about performance on that class.

Model: YOLOv8n. The original training experiment used 30 epochs, image size 640, batch size 16 and seed 42.

Test evaluation output: `MMA3001_v2_folder/model_runs/yolov8n_v2_baseline_test-6/`.

## YOLOv8s Improvement — Validation Set

The existing YOLOv8s training run recorded the following **validation** results over 119 images:

| Metric | Result |
|---|---:|
| Precision | 52.3% |
| Recall | 47.1% |
| mAP50 | 54.8% |
| mAP50–95 | 27.3% |

Model: YOLOv8s, 30 epochs, image size 640, batch size 16 and seed 42. The improvement notebook loads its saved weights and training-run confusion matrices, but does not perform a separate test evaluation. These validation metrics should not be compared directly with the YOLOv8n test results.

## Notes

Dataset cleaning uses the `.rf.` filename prefix to identify augmented images from the same source. Differently named duplicate images may not be detected. The shared `src/dataset_preparation.py` module checks the cleaned splits and records the preparation settings.
