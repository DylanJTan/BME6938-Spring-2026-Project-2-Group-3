# BME6938 Project 2: Skin Lesion Multi-Class Classification

## One-Line Summary
This project builds and compares deep learning models for five-class skin lesion image classification using the course-provided SkinLesions dataset.

## Clinical Context
Skin condition triage support can improve early identification and referral workflows. This project explores whether convolutional and transfer learning models can classify dermatologic lesion photos into five categories: acne, eksim, herpes, panu, and rosacea. This system is for educational research only and is not for clinical diagnosis.

## Project Status
Scaffold phase complete (documentation, structure, planning). Model implementation and experiments are in progress.

## Dataset
- Source path on HiPerGator: /blue/bme6938/share/Datasets/SkinLesions
- Structure: one folder per class with image files directly inside each class folder
- Classes:
	- acne
	- eksim
	- herpes
	- panu
	- rosacea

## Repository Structure
.
├── Agent_handoff.md
├── Human_handoff.md
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── docs/
│   └── Project2_Implementation_Plan.md
├── notebooks/
├── results/
│   ├── figures/
│   ├── logs/
│   └── models/
└── src/
		├── __init__.py
		├── train.py
		├── evaluate.py
		├── data/
		│   └── __init__.py
		└── models/
				└── __init__.py

## Quick Start
### 1. Environment
- Python: 3.10 or 3.11 recommended
- GPU: recommended for full training

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Paths
Update paths and experiment settings in configs/config.yaml if needed.

### 4. Run Pipeline (Scaffold Commands)
These commands are the target interface and will become fully functional as implementation progresses.
```bash
python -m src.train --config configs/config.yaml --model cnn_baseline
python -m src.train --config configs/config.yaml --model resnet18_transfer
python -m src.evaluate --config configs/config.yaml --checkpoint results/models/best.pt
```

## Expected Outputs
Generated artifacts will be stored in results/:
- models/: best and last checkpoints
- logs/: epoch metrics and run metadata
- figures/: confusion matrix, ROC (OvR), and training curves

## Evaluation Metrics
The project will report:
- Accuracy
- Precision, Recall, F1-score (per-class, macro, weighted)
- One-vs-rest ROC-AUC for multi-class
- Confusion matrix
- From-scratch vs transfer learning comparison

## Rubric Alignment Checklist
### Report (PDF)
- Abstract
- Introduction with clinical motivation and use case
- Literature review (minimum citation targets will be met in final report)
- Methods and data pipeline
- Results and evaluation with figures/tables
- Discussion, limitations, ethics, and future work

### GitHub Repository
- Public repository
- Reproducible code and meaningful commit history
- Requirements file with versions
- EDA notebook (to be added in notebooks/)
- Demo notebook (to be added in notebooks/)

## Reproducibility Notes
- Fixed random seeds will be used for data splits and training.
- Data split manifests will be persisted to avoid leakage and ensure repeatability.
- No hard-coded personal paths in training code; paths are config-driven.

## Authors and Contributions
Add team member names and roles here before submission.

## AI Usage Disclosure
AI coding assistance may be used for implementation support. Team members remain responsible for validating, understanding, and testing all code and analyses.

## License and Data Use
Dataset licensing/citation details should be added based on course-provided source information. This repository is for educational use in BME6938.