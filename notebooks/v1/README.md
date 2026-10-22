# YOLOv8n Experiments — Dataset V1

This folder contains the training, evaluation and improvement experiments for YOLOv8n using Roboflow Dataset V1.

## Notebooks

- `MMA3001_v1_training.ipynb` — Baseline YOLOv8n training.
- `MMA3001_v1_evaluation.ipynb` — Baseline model evaluation on the test set.
- `MMA3001_v1_no_mosaic.ipynb` — YOLOv8n training with mosaic augmentation disabled.
- `MMA3001_v1_checking.ipynb` — Model comparison, confidence threshold analysis and final model selection.
- `MMA3001_v1_inference.ipynb` — Package inspection using the selected no-mosaic model, including bounding-box visualisation and automated tests.
- `MMA3001_v1_learning_rate.ipynb` — AdamW learning-rate experiments and validation comparisons for Dataset V1.


## Dataset Preparation

Dataset cleaning is handled by `src/dataset_preparation.py`, which removes overlapping image sources across the training, validation and test sets and checks the annotations.

## Model Results

Model evaluation results are documented in `results/v1_evaluation_results.md`.
