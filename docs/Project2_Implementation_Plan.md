# Project 2 Implementation Plan (SkinLesions)

## Objective
Build a reproducible deep learning pipeline for five-class skin lesion classification and compare:
1. A CNN trained from scratch
2. One or more transfer learning models

Dataset root:
/blue/bme6938/share/Datasets/SkinLesions

## Phase 1: Foundation and Reproducibility
1. Finalize repository scaffold and config-driven path management.
2. Standardize command interface for train/evaluate/predict.
3. Pin dependencies and establish deterministic seed policy.
4. Add structured logging and checkpoint conventions.

## Phase 2: Data Pipeline and EDA
1. Build dataset indexer from folder names and image paths.
2. Generate deterministic stratified train/val/test split manifests.
3. Implement train augmentations and val/test transforms.
4. Create EDA notebook with:
   - Class distribution
   - Sample images
   - Resolution/aspect ratio distribution
   - Potential duplicates and quality checks
5. Document EDA-driven preprocessing choices.

## Phase 3: Modeling and Training
1. Implement custom CNN baseline (3-5 conv blocks).
2. Implement transfer model wrapper (starting with ResNet-18).
3. Support frozen-backbone warmup and fine-tuning.
4. Add:
   - Learning rate scheduler
   - Early stopping
   - Best-model checkpointing
   - Class imbalance handling (weighted loss or sampler)

## Phase 4: Evaluation and Demo
1. Compute metrics:
   - Accuracy
   - Per-class precision, recall, F1
   - Macro/weighted averages
   - OvR ROC-AUC
   - Confusion matrix
2. Save report-ready artifacts (tables/plots).
3. Compare from-scratch vs transfer learning results.
4. Build demo notebook for model loading and inference visualization.
5. Optional: add Grad-CAM examples.

## Phase 5: Documentation and Submission
1. Finalize README rubric requirements.
2. Ensure report section artifacts are ready.
3. Add authors/roles, dataset citation/licensing notes.
4. Run clean-environment reproducibility check.

## Verification Checklist
1. Class discovery matches expected five labels.
2. No split leakage between train/val/test.
3. Smoke run completes for both model families.
4. Metrics and plots are generated and versioned.
5. EDA and demo notebooks execute top to bottom.

## Risks and Mitigations
1. Class imbalance: use weighted losses/sampling and macro metrics.
2. Overfitting: data augmentation, early stopping, regularization.
3. Compute limits: prioritize ResNet-18 before heavier backbones.
4. Reproducibility drift: fixed seeds and persisted split manifests.
