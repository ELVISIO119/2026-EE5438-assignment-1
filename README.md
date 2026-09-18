# EE5438 Assignment 1 — Pure MLP Experiments

Cai Haochen · 58561440 · Semester A 2026–2027

Classify Fashion-MNIST clothing images with fully connected neural networks. Begin with the assignment's sigmoid/SGD baseline, investigate one improvement at a time, and integrate changes supported by validation results.

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

The planned sequence is: baseline → AdamW → residual blocks and normalization → Mixup → SwiGLU → mixed Muon optimization → SAM/ASAM → MLP-Mixer → EMA/SWA → MLP distillation → structured pruning/Pareto analysis → final integration. Actual results determine the final model, not the length of this list.

See [JOURNAL.md](JOURNAL.md) for dated experiment notes. Results are recorded only after a run finishes. Submission marks and bonus ranking depend on instructor assessment.
