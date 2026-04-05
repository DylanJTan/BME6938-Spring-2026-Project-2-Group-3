# Human Handoff
Update (2026-04-05):
- Legacy scaffold TODO scripts in `src/` are now functional wrappers.
- `notebooks/99_demo.ipynb` has been created.
- Training now supports class-balanced loss and transfer-backbone warmup freeze.
- `requirements.txt` now mirrors the current `pt` conda env export.
- Full benchmark runs completed with 4 models and comparison artifacts generated.

Latest test-set results (`results/logs/comparison.md`):
- cnn_baseline: accuracy 0.7399, macro F1 0.7330
- resnet18_pretrained: accuracy 0.9731, macro F1 0.9729
- resnet50_pretrained: accuracy 0.9686, macro F1 0.9685
- efficientnet_b0: accuracy 0.9955, macro F1 0.9955

Human status here:
We can start to build based off the "What Still Needs To Be Built" section below.
I have reviewed the AI status below, it makes sense.
Note that whenever you finish, I advise asking the model to update the agent handoff file for the next agent, and personally editing this file (and or using AI to help adjust it.)


Below is the AI generated status.

## Project Status
The repository has been scaffolded and documented to align with the course rubric. Core model/data code is not complete yet.

## What Is Ready
1. Rubric-oriented README draft
2. Pinned requirements (updated for conda env 'pt')
3. Config template with dataset path and training defaults
4. Written implementation plan with phases and verification checklist
5. Handoff guidance for next coding agent
6. EDA notebook at notebooks/01_eda.ipynb

## What Still Needs To Be Built
1. Fill README author/role section and literature/data citation language.
2. Select/annotate final figures/tables for the written report.
3. Draft report interpretation/discussion based on generated metrics.

## Paths To Know
1. Dataset root:
   - /blue/bme6938/share/Datasets/SkinLesions
2. Plan file:
   - docs/Project2_Implementation_Plan.md
3. Config file:
   - configs/config.yaml

## Recommended Immediate Next Steps
1. Assign one teammate to data pipeline and one to modeling.
2. Finish a smoke-run training script first before notebook polishing.
3. Start collecting literature citations now to reduce report crunch.
4. Log team member roles directly in README while work is fresh.

## Rubric Risk Watchlist
1. Missing demo notebook (notebooks/99_demo.ipynb) (required for full repository points).
2. Literature citation minimums for report still pending.
3. Need explicit dataset citation/license language in final report and README.
4. Need clear commit history from multiple teammates.
