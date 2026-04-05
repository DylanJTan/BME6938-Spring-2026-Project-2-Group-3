# Agent Handoff

## Update (2026-04-05)
1. Replaced `requirements.txt` with a direct export from conda env `pt` (`conda list -n pt --export`).
2. Filled legacy `src/` gaps by wiring:
   - `src/train.py` -> `skinlesions.scripts.train:main`
   - `src/evaluate.py` -> `skinlesions.scripts.evaluate:main`
3. Added required demo notebook:
   - `notebooks/99_demo.ipynb`
4. Implemented two pending plan items in training:
   - Class imbalance handling via `training.class_weighting` (`none` | `balanced`)
   - Transfer warmup/fine-tune via `training.freeze_backbone_epochs`
5. Added new config keys in `configs/config.yaml`:
   - `training.class_weighting`
   - `training.freeze_backbone_epochs`
6. Completed full 4-model training + evaluation benchmark and generated report artifacts.

## Completed Benchmarks (2026-04-05)
Test-split comparison from `results/logs/comparison.md`:
1. `cnn_baseline`: accuracy 0.7399, macro F1 0.7330
2. `resnet18_pretrained`: accuracy 0.9731, macro F1 0.9729
3. `resnet50_pretrained`: accuracy 0.9686, macro F1 0.9685
4. `efficientnet_b0`: accuracy 0.9955, macro F1 0.9955

Generated artifacts include:
1. `results/models/best_*.pt` and `results/models/last_*.pt` for all runs.
2. `results/logs/metrics_*.csv` (training histories).
3. `results/logs/metrics_*.json` (test metrics per checkpoint).
4. `results/logs/comparison.json` and `results/logs/comparison.md`.
5. `results/figures/confusion_matrix_*.png`, `results/figures/roc_ovr_*.png`, and `results/figures/training_curves_*.png`.

## What Has Been Completed
1. Repository scaffold created for Project 2.
2. Rubric-aligned README draft created.
3. Dependency file added with pinned versions.
4. Config template added at configs/config.yaml.
5. Master project plan added at docs/Project2_Implementation_Plan.md.
6. Starter Python entrypoint files and package layout created.
7. Updated README and requirements.txt to use conda environment 'pt' instead of pip installs.
8. Created EDA notebook at notebooks/01_eda.ipynb with class distribution, sample images, and image size analysis.

## Canonical Plan Reference
Follow this document as the source of truth:
- docs/Project2_Implementation_Plan.md

## Current State Snapshot
1. Data path confirmed: /blue/bme6938/share/Datasets/SkinLesions
2. Data layout confirmed: flat class folders, no predefined splits.
3. Environment setup updated to use conda env 'pt'.
4. EDA notebook created and ready for review.

## Priority Next Tasks (In Order)
1. Finalize README `Authors and Contributions` and citation/license text.
2. Curate best report figures (confusion matrices, ROC, training curves) from generated outputs.
3. Write report analysis comparing baseline vs transfer results and overfitting behavior.
4. Optionally run ablations using `class_weighting=balanced` and `freeze_backbone_epochs>0`.

## Required Output Contracts
1. Training should output:
   - results/models/best_*.pt
   - results/logs/*.csv or *.json
2. Evaluation should output:
   - results/figures/confusion_matrix_*.png
   - results/figures/roc_ovr_*.png
   - results/logs/metrics_*.json

## Constraints
1. Keep paths config-driven; avoid hard-coded personal directories.
2. Preserve reproducibility: deterministic seeds and saved splits.
3. Keep naming and structure consistent with README promises.
4. Prioritize correctness and rubric coverage over optional features.

## Suggested First Implementation Slice
1. src/data/dataset_index.py
2. src/data/splits.py
3. src/data/transforms.py
4. src/train.py minimal end-to-end single epoch smoke path
5. src/evaluate.py minimal confusion matrix + classification report

## Definition of Done for Next Agent Iteration
1. One smoke experiment can run end-to-end from config.
2. Split manifest and first metrics artifact are generated.
3. README quick-start commands are executable with no manual patching.
