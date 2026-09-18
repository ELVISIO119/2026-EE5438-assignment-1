# EE5438 Assignment 1 — Pure MLP Experiments

Cai Haochen · 58561440 · Semester A 2026–2027

Classify Fashion-MNIST clothing images with fully connected neural networks. Begin with the assignment's sigmoid/SGD baseline, investigate one improvement at a time, and integrate changes supported by validation results.

## Current result and files

The accuracy-first classifier achieves **94.39% test accuracy (9,439/10,000)** and **0.9438 macro F1**. Its validation-only selection score is **95.28% (5,717/6,000)**. It uses three pure MLP-Mixer checkpoints, 10/30/10 inference views, validation-calibrated probabilities and weights 0.325/0.325/0.350. The total is 2,207,556 stored parameters and 3,183,978,880 dense MACs per image. Computation is reported, not penalized during selection. No claim of a globally optimal score or guaranteed class bonus is made.

- [Executed, self-contained notebook](submission/Assign01_Cai_Haochen_58561440.ipynb)
- [ZIP with notebook and typed Section A guide](submission/Assign01_Cai_Haochen_58561440.zip)
- [Section A handwriting instructions](section_a/README.md)
- [Frozen recipe](results/final_recipe.json), [test metrics](results/final_test_metrics.json), [validation candidates](results/validation_candidates.csv)

The ZIP requires your genuine handwritten Section A PDF before Canvas submission. The included PDF is clearly labelled as a typed study guide. The notebook's default Run All unpacks its embedded code/checkpoints into a temporary directory and actually evaluates the frozen classifier. Set `RETRAIN_ALL=True` to run all training experiments from scratch. Git intentionally excludes standalone checkpoint files; the notebook contains the checkpoints needed for default evaluation.

## Rules

- Only MLP architectures, official Fashion-MNIST data, and random seed `58561440`.
- No CNN, Transformer, attention, external training images, pretrained model or out-of-scope teacher.
- Split the official training set into 54,000 training and 6,000 validation examples. Fit normalization on training images only.
- Select checkpoints, training methods, inference views and pruning settings using validation data. Freeze the final recipe before test evaluation. Treat the official test split as a benchmark; do not claim an independently blinded holdout.
- Track major ideas on branches; use commits for incremental changes; integrate measured improvements into `main`.
- Report negative results and inference costs. A named method is not automatically an improvement.
- Section A still requires the student's own handwriting. Experiments cannot satisfy that manual requirement.

## Working method

Each significant idea gets a `feature/*` branch. Implementation, checks, measurements and interpretation are separate small commits. Merge completed experiments into `main`, including negative results, so later work builds on an auditable record.

The completed sequence is baseline → AdamW → residual blocks and normalization → Mixup → SwiGLU → mixed Muon optimization → SAM/ASAM → MLP-Mixer → EMA/SWA → MLP distillation → structured pruning/Pareto analysis → regularization controls → initial integration → accuracy-first extension → expanded inference/ensemble integration. Actual results determine the final model, not the length of this list.

The final extension trains six additional recipes and retains 18 ordinary/EMA/SWA checkpoints. Across the project, 43 checkpoints were loaded and their recorded validation accuracies and physical costs recomputed. The expanded search records 506 retained candidates plus 1,776 attempted greedy weight settings. Both fixed validation halves are development data. The public test benchmark was evaluated during development; this is an exploratory continuation, not a new blinded test. Final component/temperature/weight selection uses validation only, and its hash-bound recipe precedes the final benchmark evaluation.

The experiment branches `feature/01-baseline` through `feature/13-final-integration` preserve the initial sequence. `feature/14-accuracy-first` contains the six new training recipes and incremental results; `feature/15-accuracy-integration` contains final selection, verification and delivery. Each branch remains available after merging into `main`.

See [JOURNAL.md](JOURNAL.md) for dated experiment notes. Results are recorded only after a run finishes. Submission marks and bonus ranking depend on instructor assessment.

## Run an experiment

Use Python 3.12 and the dependencies in `requirements.txt`. The recorded environment uses PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128 on an RTX 5090. CPU is supported, but the longer Mixer experiments are substantially slower and BF16 autocast is disabled there.

```bash
python3 check_models.py
python3 sharpness.py
python3 baseline.py
python3 train.py configs/adamw.json
python3 train.py configs/mixer.json
python3 train.py configs/mixer_muon.json
```

Run from the repository root. Fashion-MNIST downloads into `data/`. Every run writes its measured epoch history to `results/`, saves the validation-selected checkpoint, and appends its result to the journal. Checkpoints and dataset files are ignored by Git. Rerunning a configuration replaces its local result/checkpoint files; use a separate checkout or directory when preserving the recorded run.

`baseline.py` normalizes and flattens inputs outside its model; `train.Model` performs normalization internally. Do not normalize inputs twice when evaluating saved checkpoints. Cross-entropy takes raw logits; softmax is used for probabilities at inference, avoiding redundant softmax during training.

See [SOURCES.md](SOURCES.md) for source links and complexity-counting conventions.
