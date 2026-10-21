# Dataset V1 — YOLOv8n Evaluation Results

## Dataset

Roboflow Dataset V1 contains four defect classes: loose-meat, packaging-error, twisted-meat and unsealed. Source overlap was removed before model evaluation.

| Split | Images |
|:--|--:|
| Training | 1,935 |
| Validation | 119 |
| Test | 80 |

The test set contains **99 labelled defect instances** across 80 images.

## Test-set Comparison

| Model | Precision | Recall | mAP50 | mAP50–95 |
|:--|--:|--:|--:|--:|
| YOLOv8n baseline | 72.1% | 51.8% | 58.2% | 24.1% |
| YOLOv8n no mosaic | 52.8% | 52.4% | 54.4% | 22.6% |

The no-mosaic model has slightly higher overall test recall (52.4% versus 51.8%), but lower precision and mAP50. For packaging-error, test recall is 45.9% for no mosaic and 35.1% for baseline. The comparison uses the same 80-image test split.

**Evaluation settings:** image size 640; the no-mosaic test evaluation uses `conf=0.001` and `iou=0.7`. The baseline test evaluation uses Ultralytics' default validation confidence settings. The model's inspection function instead uses `conf=0.201`; the tabulated test metrics should not be interpreted as measurements at that inspection threshold.

## Validation-set Comparison

| Model | Precision | Recall | mAP50 | mAP50–95 |
|:--|--:|--:|--:|--:|
| YOLOv8n baseline | 77.6% | 46.8% | 68.3% | 32.3% |
| YOLOv8n no mosaic | 64.1% | 62.3% | 69.5% | 32.1% |

Both models were evaluated on 119 validation images (154 labelled instances) with `conf=0.001` and `iou=0.7`.

## Limitations

Source overlap is checked using Roboflow `.rf.` filename identifiers, so differently named duplicates may not be detected. The test set contains only seven twisted-meat instances, which limits conclusions for that class. Validation-set comparisons and test-set evaluation are reported separately.

Learning-rate tuning is documented separately in `MMA3001_v1_learning_rate.ipynb`.
