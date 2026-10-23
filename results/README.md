# Results

This folder contains model evaluation results and automated test reports for the pork rasher packaging defect detection project.

## Model Evaluation

| File | Description |
|---|---|
| `v4_baseline_evaluation_results.md` | YOLOv8n baseline evaluation using Dataset V4. |
| `v2_evaluation_results.md` | YOLOv8n test results and YOLOv8s validation results using Dataset V2. |
| `v1_evaluation_results.md` | Dataset V1 baseline and no-mosaic model comparisons, including validation and test results. |

## Automated Testing

- `test_report.txt` — Results of the six automated dataset preparation tests.

The inspection function tests are documented separately in the repository root under `test_report.txt`.

## Notes

Model performance is evaluated using precision, recall, mAP50 and mAP50–95. Validation results are used for model comparison and tuning, while test results provide a separate assessment of model performance.

Dataset V1, V2 and V4 contain different annotations and class distributions, so their results should not be treated as directly comparable.
