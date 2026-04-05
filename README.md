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
- Dataset access: course-provided storage path above (not redistributed in this repo)
- The dataset is originally from the MedMNIST Zenodo and group, in this case the DermaMNIST set that can be accessed at https://github.com/MedMNIST/MedMNIST
- To recreate, edit all paths for code to your data path that stores the DermaMNIST dataset.
- License/citation: Refer to the bottom of this README.md

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
│       ├── run_full.py           # Default CLI: splits + train + compare eval
│       ├── generate_figures.py   # Default CLI: figures/logs from saved checkpoints
│       ├── make_splits.py        # Low-level split utility
│       ├── train.py              # Low-level training utility
│       └── evaluate.py           # Low-level evaluation utility
├── results/
│   ├── figures/             # Confusion matrices, ROC curves, training curves
│   ├── logs/                # Epoch metrics CSV, per-model JSON, comparison
│   │   └── splits/          # train.csv, val.csv, test.csv manifests
│   └── models/              # best_*.pt and last_*.pt checkpoints
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 99_demo.ipynb
├── docs/
│   ├── Project2_Implementation_Plan.md
│   ├── Submission_Checklist.md
│  
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip-style package list (reference)
└── setup.py                 # Editable install
```

## Quick Start (HiPerGator)

### Runtime & Compute
- Python: 3.12 (tested)
- Recommended hardware: NVIDIA GPU with >= 8GB VRAM for transfer-learning runs
- Typical runtime on A100-class GPU:
    - Full pipeline (`run_full`) with 4 models: roughly 10-20 minutes (depends on queue/load)
    - Figures-only regeneration: under 1 minute for existing checkpoints
- Disk usage: checkpoints + figures + logs typically a few hundred MB

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
pip install -r requirements.txt
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

### 4. Reproduce Everything (Default)

```bash
python -m skinlesions.scripts.run_full --config configs/config.yaml
```

This single command runs the full pipeline end-to-end:
- deterministic split generation
- training across the default model set
- test-set evaluation, figures, and comparison tables

### 5. Generate Figures/Logs Only (No Training)

```bash
python -m skinlesions.scripts.generate_figures --config configs/config.yaml
```

This command auto-discovers `best_*.pt` checkpoints in `results/models/` and regenerates:
- confusion matrices
- ROC curves
- per-model metrics JSON files
- `comparison.json` and `comparison.md`

### 5.1 Demo Notebook

Run the demo notebook top-to-bottom to load a trained checkpoint and visualize predictions:
- `notebooks/99_demo.ipynb`

### 6. Low-Level Commands (Optional)

If you want fine-grained control over each stage, use the low-level scripts below.

```bash
# 1) Create split manifests only
python -m skinlesions.scripts.make_splits --config configs/config.yaml

# 2) Train one model only
python -m skinlesions.scripts.train \
    --config configs/config.yaml \
    --model resnet18_pretrained \
    --experiment-name resnet18_run

# 3) Evaluate one or more checkpoints
python -m skinlesions.scripts.evaluate \
    --config configs/config.yaml \
    --checkpoint results/models/best_resnet18_pretrained_resnet18_run.pt
```

### 7. Previous Per-Model Training Examples

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

### 8. Evaluate & Compare

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

## Results Summary (Current)

Latest test-set comparison (from `results/logs/comparison.md`):

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | ROC-AUC (Macro) |
|---|---:|---:|---:|---:|
| cnn_baseline | 0.7399 | 0.7330 | 0.7332 | 0.9300 |
| resnet18_pretrained | 0.9731 | 0.9729 | 0.9730 | 0.9991 |
| resnet50_pretrained | 0.9686 | 0.9685 | 0.9685 | 0.9990 |
| efficientnet_b0 | 0.9955 | 0.9955 | 0.9955 | 1.0000 |

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
- Demo notebook (`notebooks/99_demo.ipynb`)
- End-to-end scripts

## Authors and Contributions
All members contributed to the report itself equally.
- Member 1: Dylan Tan - Training/Eval/Results/Init
- Member 2: Lauren Plummer - EDA, Literature Review
- Member 3: Riley Bendure — Training/Eval, Literature Review

## AI Usage Disclosure
AI coding assistance was used for implementation support. Team members remain responsible for validating, understanding, and testing all code and analyses.

## License and Data Use
This repository is for educational use in BME6938.

The dataset has been retrieved from the MedMNIST Zenodo and group.
Citations below:
Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

License:
The MedMNIST dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), except DermaMNIST under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The code is under Apache-2.0 License.

We use the DermaMnist set in particular, original paper citations/acknowledgment:
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
