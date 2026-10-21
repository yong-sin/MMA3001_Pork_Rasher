# YOLOv8n Baseline Evaluation on Dataset V4

Results recorded from `MMA3001_02_Evaluation.ipynb` (saved Colab output).

- Test images: 80 (12 background images)
- Labelled defect instances: 79
- Precision (mean): 0.5646 (56.46%)
- Recall (mean): 0.4902 (49.02%)
- mAP50: 0.5333 (53.33%)
- mAP50-95: 0.2659 (26.59%)

The cleaned test split contains 3 loose-meat, 23 packaging-error, 4 twisted-meat, 49 unsealed, and 0 wrinkle annotations. The test data therefore cannot measure wrinkle recall or AP.

The cleaned dataset split matched the saved preparation manifest. The notebook loaded existing baseline `best.pt` weights, but the preparation manifest alone does not verify the original training history of those weights.

Evaluation plots were saved in Colab/Google Drive under `model_runs/yolov8n_baseline_test-6`; these plots are not bundled here.
