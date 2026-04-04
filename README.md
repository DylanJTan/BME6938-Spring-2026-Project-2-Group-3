# BME6938 Project 2: Skin Lesion Multi-Class Classification

## One-Line Summary
This project builds and compares deep learning models for five-class skin lesion image classification using the course-provided SkinLesions dataset.

## Clinical Context
Skin condition triage support can improve early identification and referral workflows. This project explores whether convolutional and transfer learning models can classify dermatologic lesion photos into five categories: acne, eksim, herpes, panu, and rosacea. This system is for educational research only and is not for clinical diagnosis.

## Project Status
Full pipeline implemented. Run the commands below to reproduce all experiments end-to-end.

## Dataset
- Source path on HiPerGator: `/blue/bme6938/share/Datasets/SkinLesions`
- Structure: one folder per class with image files directly inside each class folder
- Classes: `acne`, `eksim`, `herpes`, `panu`, `rosacea`

## Repository Structure
```
.
├── configs/
│   └── config.yaml          # Central experiment configuration
├── skinlesions/             # Main Python package
│   ├── __init__.py
│   ├── transforms.py        # Train/val/test augmentation pipelines
│   ├── models.py            # CNN baseline + transfer learning models
│   ├── train_utils.py       # Seed, training loops, AMP, early stopping
│   ├── metrics.py           # Accuracy, precision, recall, F1, ROC-AUC
│   ├── plots.py             # Confusion matrix & ROC curve figures
│   ├── data/
│   │   ├── dataset.py       # CSV-manifest-backed PyTorch Dataset
│   │   ├── splits.py        # Stratified split utilities
│   │   └── loader.py        # DataLoader builder
│   └── scripts/
│       ├── make_splits.py   # CLI: create split manifests
│       ├── train.py         # CLI: full training loop
│       └── evaluate.py      # CLI: test-set evaluation & comparison
├── src/                     # Legacy scaffold (entrypoint stubs)
├── results/
│   ├── figures/             # Confusion matrices, ROC curves, training curves
│   ├── logs/                # Epoch metrics CSV, per-model JSON, comparison
│   │   └── splits/          # train.csv, val.csv, test.csv manifests
│   └── models/              # best_*.pt and last_*.pt checkpoints
├── notebooks/
│   └── 01_eda.ipynb
├── docs/
│   └── Project2_Implementation_Plan.md
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip-style package list (reference)
└── setup.py                 # Editable install
```

## Quick Start (HiPerGator)

### 1. Environment

```bash
# Use the pre-built conda environment
conda activate pt

# OR recreate from environment.yml on a new machine
conda env create -f environment.yml -n pt
conda activate pt
```

### 2. Install the package (editable)

```bash
# From the repository root
pip install -e .
```

### 3. Configure Paths

All paths are set in `configs/config.yaml`.  The dataset root is already
configured for HiPerGator:

```yaml
paths:
  dataset_root: /blue/bme6938/share/Datasets/SkinLesions
```

Adjust `project.experiment_name` to label your run.

### 4. Create Data Split Manifests

```bash
python -m skinlesions.scripts.make_splits --config configs/config.yaml
```

Manifests are written to `results/logs/splits/` (train.csv, val.csv, test.csv).
Re-running with the same seed produces identical splits (deterministic stratified sampling, seed=42).

### 5. Train Models

```bash
# Baseline CNN (trained from scratch)
python -m skinlesions.scripts.train \
    --config configs/config.yaml \
    --model cnn_baseline \
    --experiment-name baseline_run

# Transfer learning – ResNet-18
python -m skinlesions.scripts.train \
    --config configs/config.yaml \
    --model resnet18_pretrained \
    --experiment-name resnet18_run

# Transfer learning – ResNet-50
python -m skinlesions.scripts.train \
    --config configs/config.yaml \
    --model resnet50_pretrained \
    --experiment-name resnet50_run

# Transfer learning – EfficientNet-B0
python -m skinlesions.scripts.train \
    --config configs/config.yaml \
    --model efficientnet_b0 \
    --experiment-name effb0_run
```

Checkpoints are saved to `results/models/best_<model>_<experiment>.pt`
and `results/models/last_<model>_<experiment>.pt`.
Epoch metrics are logged to `results/logs/metrics_<run_name>.csv`.

Training uses mixed-precision AMP (when CUDA is available) and AdamW with
cosine annealing LR schedule. Early stopping halts training when validation
loss does not improve for 10 consecutive epochs.

### 6. Evaluate & Compare

```bash
# Single model on the test split
python -m skinlesions.scripts.evaluate \
    --config configs/config.yaml \
    --checkpoint results/models/best_cnn_baseline_baseline_run.pt

# Compare multiple models in one call
python -m skinlesions.scripts.evaluate \
    --config configs/config.yaml \
    --checkpoint \
        results/models/best_cnn_baseline_baseline_run.pt \
        results/models/best_resnet18_pretrained_resnet18_run.pt \
    --summary-out results/logs/comparison.json
```

Evaluation outputs (per checkpoint):
- `results/figures/confusion_matrix_<tag>.png`
- `results/figures/roc_ovr_<tag>.png`
- `results/logs/metrics_<tag>.json`

When multiple checkpoints are passed:
- `results/logs/comparison.json`
- `results/logs/comparison.md` (markdown table)

## Models

| Key | Description |
|-----|-------------|
| `cnn_baseline` | Custom 5-block CNN trained from scratch |
| `resnet18_pretrained` | ResNet-18 with ImageNet weights, FC replaced |
| `resnet50_pretrained` | ResNet-50 with ImageNet weights, FC replaced |
| `efficientnet_b0` | EfficientNet-B0 with ImageNet weights, classifier replaced |

## Expected Outputs
Generated artifacts are stored in `results/`:
- `models/`: best and last checkpoints (`.pt`)
- `logs/`: epoch metrics CSV, per-model JSON metrics, comparison JSON/MD
- `logs/splits/`: deterministic train/val/test manifests
- `figures/`: confusion matrix, ROC (OvR), and training curves

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (macro, weighted, and per-class)
- One-vs-rest ROC-AUC per class and macro average
- Confusion matrix (normalised heatmap)
- Cross-model comparison table

## Reproducibility Notes
- All experiments seeded with `seed: 42` (Python, NumPy, PyTorch, CUDA).
- Split manifests are persisted to `results/logs/splits/` to prevent leakage.
- Re-run `make_splits` with the same seed to reproduce identical splits.
- No hard-coded personal paths; all paths are driven by `configs/config.yaml`.

## Rubric Alignment Checklist
### Report (PDF)
- Abstract
- Introduction with clinical motivation and use case
- Literature review
- Methods and data pipeline
- Results and evaluation with figures/tables
- Discussion, limitations, ethics, and future work

### GitHub Repository
- Public repository with reproducible code
- Requirements file with versions
- EDA notebook (`notebooks/01_eda.ipynb`)
- End-to-end demo scripts

## Authors and Contributions
Add team member names and roles here before submission.

## AI Usage Disclosure
AI coding assistance was used for implementation support. Team members remain responsible for validating, understanding, and testing all code and analyses.

## License and Data Use
Dataset licensing/citation details should be added based on course-provided source information. This repository is for educational use in BME6938.
