# Agent Handoff

## What Has Been Completed
1. Repository scaffold created for Project 2.
2. Rubric-aligned README draft created.
3. Dependency file added with pinned versions.
4. Config template added at configs/config.yaml.
5. Master project plan added at docs/Project2_Implementation_Plan.md.
6. Starter Python entrypoint files and package layout created.

## Canonical Plan Reference
Follow this document as the source of truth:
- docs/Project2_Implementation_Plan.md

## Current State Snapshot
1. Data path confirmed: /blue/bme6938/share/Datasets/SkinLesions
2. Data layout confirmed: flat class folders, no predefined splits.
3. Core code implementation is still pending.

## Priority Next Tasks (In Order)
1. Implement dataset indexing and stratified split generator.
2. Implement transform pipelines for train/val/test.
3. Build custom CNN baseline and ResNet-18 transfer wrapper.
4. Implement training loop with scheduler, early stopping, checkpointing.
5. Implement evaluation metrics and plot generation.
6. Create notebooks/01_eda.ipynb and notebooks/99_demo.ipynb.

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
