# Pork Rasher Packaging Defect Detection
A computer-vision project investigating how YOLOv8 object detection can identify visible packaging and meat-arrangement defects in images of pork rashers. The repository contains experiments on Roboflow Dataset V1, V2 and V4, reusable Python functions, model evaluations, automated tests and documentation.

> This is an academic proof of concept, not a validated industrial inspection system.

## Project overview
Packaging inspection can help identify products that may need further quality checks. This project tests whether an image-based model can locate and classify visible defects, and examines how model configuration affects precision, recall and detection performance. 
| Item | Description |
|---|---|
| **Input** | An RGB image of pork rasher packaging. Training and evaluation additionally use YOLO-format image annotations and a `data.yaml` file. |
| **Output** | Predicted defect class, confidence (0–1) and bounding box `[x1, y1, x2, y2]` in pixel coordinates; the inference notebook also shows an annotated image. |
| **Method** | Ultralytics YOLOv8 object detection, with comparisons involving mosaic augmentation, learning rates, input resolution and model size. |
| **Main objective** | Evaluate the strengths, weaknesses and practical limitations of defect detection on the available labelled images. |

The defect categories depend on the Roboflow dataset version. V1 and V2 contain loose-meat, packaging-error, twisted-meat and unsealed; V4 additionally includes wrinkle.

## Repository structure
| If you want to… | Open… |
|---|---|
| Start with the current V4 baseline | [`notebooks/baseline/`](notebooks/baseline/) |
| Review V1 baseline, no-mosaic and tuning work | [`notebooks/v1/`](notebooks/v1/) |
| Compare YOLOv8n and YOLOv8s on V2 | [`notebooks/v2/`](notebooks/v2/) |
| Understand dataset cleaning | [`src/dataset_preparation.py`](src/dataset_preparation.py) |
| Run an image inspection | [`notebooks/v1/MMA3001_v1_inference.ipynb`](notebooks/v1/MMA3001_v1_inference.ipynb) or [`inspection.py`](inspection.py) |
| Review numerical results | [`results/`](results/) |
| Check automated tests | [`tests/`](tests/) and [`test_inspection.py`](test_inspection.py) |
| Read descriptions of Python functions | [`docs/`](docs/) |

Each notebook subfolder also has a README.md briefly explaining its experiments.

## Original dataset and preparation
Dataset credit: Hello, Pork Rasher Error (Packaging), hosted on Roboflow Universe.
- [Original Pork Rasher Error (Packaging) dataset — Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/browse?queryText=sort%3Afilename&pageSize=100&startingIndex=0&browseQuery=true) 
- Dataset versions used: V1, V2 and V4. Select the required version on Roboflow and export it in a compatible YOLO format; the project uses the corresponding exported ZIP files.
- The original images, annotations and dataset rights belong to their respective owners. Check Roboflow's dataset licence and access conditions before redistribution or reuse. The repository's MIT licence does not override the dataset's terms.

### How dataset preparation works
The common [`src/dataset_preparation.py`](src/dataset_preparation.py) module is reused across the dataset versions. It:
1. Extracts the original Roboflow ZIP and locates the YOLO dataset and data.yaml.
2. Uses the filename prefix before .rf. to identify images derived from the same source, removing cross-split overlap with priority test → validation → training.
3. Checks image/label pairs, class identifiers and YOLO bounding-box coordinates.
4. Writes a preparation manifest containing settings, counts and fingerprints for consistency checks.

Preparation starts from the source ZIP, so the cleaned dataset can be recreated without manually moving images. Some experiment-specific holdout settings may differ; consult the relevant notebook. The source-ID approach recognises JPG files and cannot guarantee that differently named visual duplicates have been removed.

Dataset versions and annotations are not interchangeable. Results on V1, V2 and V4 should not be treated as a direct same-test-set ranking.

## Trained model weights, 9 training runs
[Download the trained best.pt checkpoints from Google Drive
](https://drive.google.com/drive/folders/1oqgTlR9auHvHXrplmm-v_Ol42O0HKylg?usp=sharing)

The shared folder is organised for the nine training configurations covered by the project. Each model's best.pt is the checkpoint selected by its training run; it does not mean that the model is the best performer across every experiment.
| Dataset | Training configuration | Purpose |
|---|---|---|
| V4 | YOLOv8n baseline | Dataset V4 baseline |
| V2 | YOLOv8n baseline | Dataset V2 baseline |
| V2 | YOLOv8s experiment | Model-size comparison |
| V1 | YOLOv8n baseline, 640 px | Baseline comparison |
| V1 | YOLOv8n no mosaic, 640 px | **Selected inspection demonstration model** |
| V1 | YOLOv8n no mosaic, 960 px | Input-resolution experiment |
| V1 | YOLOv8n no mosaic, AdamW LR 0.00125, 30 epochs | Learning-rate experiment |
| V1 | YOLOv8n no mosaic, AdamW LR 0.0005, 30 epochs | Learning-rate experiment |
| V1 | YOLOv8n no mosaic, AdamW LR 0.0005, 50 epochs | Learning-rate/epoch experiment |

Download the checkpoint matching the notebook you are running and update its weights/best.pt path. The notebook paths originally pointed to Google Drive folders used during development, so they are not automatically valid on another account. 

### Selected model for the inspection demonstration
The [`MMA3001_v1_inference.ipynb`](notebooks/v1/MMA3001_v1_inference.ipynb) demonstration uses the V1 YOLOv8n no-mosaic model at 640 px with a confidence threshold of 0.201. This is the selected model for the inspection demonstration, not a claim that it is the best model on all metrics.
Validation results used to compare the V1 configurations (119 images, 154 labelled instances):
| Metric | V1 baseline | V1 no mosaic (640 px) |
|---|---:|---:|
| Precision | **77.6%** | 64.1% |
| Recall | 46.8% | **62.3%** |
| mAP50 | 68.3% | **69.5%** |
| mAP50–95 | **32.3%** | 32.1% |

The no-mosaic configuration showed **higher validation recall** and a small improvement in validation mAP50, making it a useful candidate for an inspection demonstration focused on identifying potentially defective packages. This came with lower precision, so the trade-off matters.

The **independent V1 test results** were less favourable to no-mosaic: the baseline achieved **58.2% mAP50** compared with **54.4%** for no-mosaic and had much higher precision (**72.1%** versus **52.8%**). No-mosaic had slightly higher test recall (**52.4%** versus **51.8%**), including packaging-error recall of **45.9%** versus **35.1%**. The test evaluations used different confidence settings, so the test results should be interpreted cautiously rather than used to claim a definitive fair ranking.

**Selection and evaluation are separate:** Validation results informed the comparison of settings. Independent test results are reported to show their generalisation limits, rather than being used to select whichever model scored highest on the test set. The inference threshold of 0.201 is separate from the thresholds used to calculate the tabled evaluation metrics.

**Industrial-use limitation:** This is an academic proof of concept, not a validated industrial quality-control system. Limited data for some classes, false detections, missed defects, and untested changes in lighting, cameras and packaging conditions would need to be addressed before deployment. Larger independent production datasets and application-specific acceptance criteria would be needed.

## Installation and setup
The notebooks were developed primarily in Google Colab. A GPU is useful for model training; dataset tests, documentation and using existing checkpoint files do not require repeating training.
### Obtain the repository and dependencies
From a terminal:
```bash
git clone https://github.com/yong-sin/MMA3001_Pork_Rasher.git
cd MMA3001_Pork_Rasher
python -m pip install -r requirements.txt
```
The [`requirements.txt`](requirements.txt) file lists the project packages, including Ultralytics, PyTorch, NumPy, Pandas, Matplotlib, Pillow, PyYAML, pytest and pdoc. Installation of GPU-enabled PyTorch depends on the environment. If the repository is private, cloning requires authorised GitHub access.

For Google Colab, use **!** before shell commands (for example, !python -m pip install -r requirements.txt).

### Prepare files before running notebooks
1. Download the required Roboflow dataset version (V1, V2 or V4) in YOLO format.
2. Download the matching best.pt checkpoint from the shared model folder for evaluation or inference.
3. Open the relevant notebook in Colab and check PROJECT_ROOT, dataset ZIP paths, RUNS_DIR and checkpoint paths. Put your files where those paths point or update the paths to your own directories.
4. Ensure src/dataset_preparation.py is available to the notebook. The notebooks rely on the shared preparation module rather than containing separate copies of its cleaning logic.
The reorganised training notebooks keep retraining **disabled by default**, using flags such as RETRAIN_BASELINE, RETRAIN_NO_MOSAIC, RETRAIN_YOLOV8S or TRAIN_MODELS. Check the specific notebook before enabling training. A fresh training run may produce different results, even with similar settings.

## Exactly which notebook should I run?
| Task | Notebook(s) |
|---|---|
| Prepare/train the V4 baseline | [`V4 Training`](notebooks/baseline/MMA3001_v4_Training.ipynb) |
| Evaluate the V4 baseline | [`V4 Evaluation`](notebooks/baseline/MMA3001_v4_Evaluation.ipynb) |
| Train/load the V2 YOLOv8n baseline | [`V2 Training`](notebooks/v2/MMA3001_v2_training.ipynb) |
| Evaluate the V2 YOLOv8n baseline | [`V2 Validation`](notebooks/v2/MMA3001_v2_validation.ipynb) |
| Study the V2 YOLOv8s experiment | [`V2 Improvement`](notebooks/v2/MMA3001_v2_improvement.ipynb) |
| Train/load the V1 baseline | [`V1 Training`](notebooks/v1/MA3001_v1_training.ipynb) |
| Evaluate the V1 baseline | [`V1 Evaluation`](notebooks/v1/MMA3001_v1_evaluation.ipynb) |
| Explore the V1 no-mosaic model | [`V1 No Mosaic`](notebooks/v1/MMA3001_v1_no_mosaic.ipynb) |
| Compare V1 learning rates | [`V1 Learning Rate`](notebooks/v1/MMA3001_v1_learning_rate_exploration.ipynb) |
| Compare V1 models, thresholds and final test behaviour | [`V1 Checking`](notebooks/v1/MMA3001_v1_checking.ipynb) |
| Display an input image and its annotated detection | [`V1 Inference`](notebooks/v1/MMA3001_v1_inference.ipynb) |
| Investigate input-resolution effects | [`V1 Resolution_960`](notebooks/v1/MMA3001_v1_resolution_960.ipynb) |
   

Use the existing weights to reproduce evaluations without repeating 30–50 epochs of training. For a new training experiment, use the appropriate training notebook and explicitly enable training after configuring the dataset and GPU runtime.

## Run inference from Python
From the repository root, with a downloaded checkpoint and a JPG image:
```Python
from inspection import inspect_package

detections = inspect_package(
    image_path="path/to/package.jpg",
    weights_path="path/to/v1_no_mosaic_640_best.pt",
    confidence=0.201,
)
print(detections)
```
The function returns a list of dictionaries containing defect, confidence and box. The [V1 inference notebook](notebooks/v1/MMA3001_v1_inference.ipynb) also displays the original and annotated images. Its example demonstrates the function, rather than measuring model accuracy on an independent population.

## Evaluation results
**Validation results** are used to compare configurations and investigate settings such as mosaic augmentation, learning rate and model size. Independent test results provide a separate assessment of detection performance. The V1 baseline-versus-no-mosaic validation comparison is shown in Selected model for the inspection demonstration.
### Independent test-set metrics
| Dataset/model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| V1 YOLOv8n baseline | 72.1% | 51.8% | 58.2% | 24.1% |
| V1 YOLOv8n no mosaic | 52.8% | 52.4% | 54.4% | 22.6% |
| V2 YOLOv8n baseline | 77.52% | 43.92% | 49.85% | 24.34% |
| V4 YOLOv8n baseline | 56.46% | 49.02% | 53.33% | 26.59% |

The V1 baseline has stronger overall V1 test metrics than no-mosaic apart from recall. However, the baseline and no-mosaic test evaluations did not use identical confidence settings; see the [V1 results report](results/v1_evaluation_results.md) for the settings and class-level findings. Scores across V1, V2 and V4 are not direct head-to-head comparisons because dataset versions and annotations differ.

The V2 YOLOv8s experiment recorded 54.8% validation mAP50, with no separately documented independent YOLOv8s test evaluation. This must not be compared as though it were a test-set metric.

Detailed results, experimental settings and limitations:
- [V1 evaluation results](results/v1_evaluation_results.md)
- [V2 evaluation results](results/v2_evaluation_results.md)
- [V4 baseline evaluation results](results/v4_baseline_evaluation_results.md)

## Automated tests and API documentation
Run the source-code tests from the repository root:
```bash
python -m pytest -q test_inspection.py tests/test_dataset_preparation.py
```
- [`test_inspection.py`](test_inspection.py) checks the inspection function with mocked zero-detection and one-detection predictions (2 tests).
- [`tests/test_dataset_preparation.py`](tests/test_dataset_preparation.py) checks dataset cleaning, annotation validation and preparation consistency (6 tests).
- Results are recorded in [`test_report.txt`](test_report.txt) and [`results/test_report.txt`](results/test_report.txt).
These are software unit tests, not measures of YOLO detection accuracy. The actual detection metrics come from held-out evaluation.

The [`docs/`](docs/) folder contains [`inspection.html`](docs/inspection.html), [`dataset_preparation.html`](docs/dataset_preparation.html) and an [`index.html`](docs/index.html) guide to the source-code functions, parameters, returns and implementation. Source code and docstrings should be treated as authoritative if documentation needs updating.

## Assumptions, acknowledgements and licence
The experiments are limited by the available Roboflow labels and dataset splits. The .rf. filename rule can miss differently named visual duplicates; rare defect classes have few examples; and model performance has not been independently validated under production conditions. For example, Dataset V4's test set has no labelled wrinkle examples, so its test metrics cannot establish wrinkle recall or average precision.

Detection confidence is not a calibrated measure of defect certainty. Any practical system would need additional datasets, checks for false alarms and missed defects, and validation across different cameras, lighting conditions and packaging types. The reported results should not be used to make autonomous industrial acceptance or rejection decisions.

## Project Report
The complete project report is available here: [View MMA3001 Project Report (PDF)](docs/MMA3001_Project_Report.pdf)

## AI assistance and verification
ChatGPT provided substantial guidance during coding and documentation, particularly when structuring and revising Python and Colab code, explaining YOLOv8 settings, checking implementation logic, identifying potential bugs, and interpreting errors. It also assisted with dataset-cleaning workflows, organising GitHub folders and commits, drafting concise comments and docstrings, discussing limitations, and improving the clarity and professionalism of the README and other documentation.

AI was used to help work through unfamiliar tasks and troubleshoot problems, not as evidence that code or models had been validated. Suggestions were checked against the actual notebooks, dataset audits, saved training and evaluation outputs, and automated test reports. Examples of corrections included resolving Google Drive path issues between accounts and clarifying which figures were validation results versus independent test results. Code changes needed to be reviewed and run in the project environment; a successful syntax check or a mocked unit test alone did not establish model accuracy.

**Project responsibility:** Dataset choices, experimental decisions, running models in Colab, inspecting results, deciding what to retain, and interpreting the engineering trade-offs remained the student's (my) responsibility. AI assistance was most significant in programming guidance, debugging and documentation; the reported numerical metrics come from the project's recorded computations, not estimates supplied by AI. The separate project report provides a fuller reflection on the benefits, limitations and verification of AI use.

## Dataset acknowledgement and licence
**Original dataset:** [Hello — Pork Rasher Error (Packaging), Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/browse?queryText=sort%3Afilename&pageSize=100&startingIndex=0&browseQuery=true). Roboflow and Ultralytics provide dataset hosting/export and YOLOv8 software respectively. These external resources are subject to their own terms and licences.

Repository code licence: MIT. The repository's MIT licence does not automatically apply to the original dataset or third-party model assets.
