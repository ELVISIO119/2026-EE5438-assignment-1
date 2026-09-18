# EE5438 Assignment 1 — Pure MLP Experiments

Cai Haochen · 58561440 · Semester A 2026–2027

Classify Fashion-MNIST clothing images with fully connected neural networks. Begin with the assignment's sigmoid/SGD baseline, investigate one improvement at a time, and integrate changes supported by validation results.

## Current result and files

The frozen classifier achieves **94.42% test accuracy (9,442/10,000)** and **0.9440 macro F1**. Its validation-only selection score is **95.35% (5,721/6,000)**. A confidence gate first runs the wide Mixer on one view. Predictions of Trouser, Sandal, Sneaker, Bag or Ankle boot with probability at least 0.90 exit early; the remaining images use the three-model ensemble with 10/30/10 views, calibrated probabilities and weights 0.325/0.325/0.350. The gate uses predictions, never true labels, and shares existing weights.

There are **2,207,556 stored parameters**, unchanged from the ungated reference. On the test set, 4,913/10,000 images exit early and average dense MACs fall from 3,183,978,880 to **1,696,912,840 per image (46.7% lower)**. Worst-case MACs rise slightly to **3,261,201,664** because a hard image incurs both stages. Measured on the same 1,024 validation inputs, inference takes 235.9 ms versus 388.7 ms: **1.65x throughput** on the shared RTX 5090. These are conditional average costs and descriptive timings, not a guarantee of lower worst-case FLOPs or a class tie-break award. Selection prioritizes accuracy, then lower cost for ties; no global optimum or top-ten ranking is claimed.

- [Executed, self-contained notebook](submission/Assign01_Cai_Haochen_58561440.ipynb)
- [ZIP with notebook and typed Section A guide](submission/Assign01_Cai_Haochen_58561440.zip)
- [Section A handwriting instructions](section_a/README.md)
- [Frozen recipe](results/final_recipe.json), [test metrics](results/final_test_metrics.json), [cascade candidates](results/cascade_candidates.csv), [cascade timing](results/cascade_benchmark.json)
- [NVFP4 measurements](results/nvfp4_benchmark.json), [garment trials](results/garment_candidates.csv), [spatial-readout trials](results/spatial_candidates.csv)

The ZIP requires your genuine handwritten Section A PDF before Canvas submission. The included PDF is clearly labelled as a typed study guide. The notebook's default Run All unpacks its embedded code/checkpoints into a temporary directory and actually evaluates the frozen classifier. Set `RETRAIN_ALL=True` to run all training experiments from scratch. Git intentionally excludes standalone checkpoint files; the notebook contains the checkpoints needed for default evaluation.

## Rules

- Only MLP architectures, official Fashion-MNIST data, and random seed `58561440`.
- No CNN, Transformer, attention, external training images, externally pretrained weights or out-of-scope teacher.
- Split the official training set into 54,000 training and 6,000 validation examples. Fit normalization on training images only.
- Select checkpoints, training methods, inference views and pruning settings using validation data. Freeze the final recipe before test evaluation. Treat the official test split as a benchmark; do not claim an independently blinded holdout.
- Track major ideas on branches; use commits for incremental changes; integrate measured improvements into `main`.
- Report negative results and inference costs. A named method is not automatically an improvement.
- Section A still requires the student's own handwriting. Experiments cannot satisfy that manual requirement.

## Working method

Each significant idea gets a `feature/*` branch. Implementation, checks, measurements and interpretation are separate small commits. Merge completed experiments into `main`, including negative results, so later work builds on an auditable record.

The completed sequence is baseline → AdamW → residual blocks and normalization → Mixup → SwiGLU → mixed Muon optimization → SAM/ASAM → MLP-Mixer → EMA/SWA → MLP distillation → structured pruning/Pareto analysis → regularization controls → initial integration → accuracy-first extension → expanded inference/ensemble integration → targeted garment/focal losses → real NVFP4 benchmark → spatial readouts → confidence-gated inference. Actual results determine the final model, not the length of this list.

The accuracy-first extension trained six recipes and retained 18 ordinary/EMA/SWA checkpoints. The garment/spatial follow-up added eight training recipes and 24 checkpoints, bringing that stage to 67 checkpoints. The detector-inspired follow-up below adds another 24; all **91 checkpoints** and the final routed validation pipeline have now been verified. The initial expanded search records 506 retained candidates plus 1,776 attempted greedy weight settings. The two refinement rounds each retain 583 validation candidates, and the gate search retains 43. Both fixed validation halves are development data. The public test benchmark was evaluated during development; the garment hypothesis follows earlier public test errors, so this is exploratory continuation, not a new blinded test. New fitting and selection use training/validation only, and the hash-bound gate recipe precedes its final benchmark evaluation.

Targeted losses and spatial readouts did not displace the reference ensemble. **Shirt test recall remains 81.4% and F1 remains 0.8285**: this bottleneck is not solved by the current follow-up. The new gate changes 16 test predictions, fixes eight errors and introduces five, for a net gain of three images. This small change is not presented as statistically significant. The main demonstrated gain is average inference efficiency with slightly higher observed benchmark accuracy.

NVFP4 used real packed W4A4 channel matrices and a profiled SM120 FP4 kernel. On the ungated reference, eager NVFP4 reduced validation accuracy from 95.2833% to 94.9000% and increased batch-128 latency from 49.95 ms to 97.63 ms. With both paths compiled, original precision took 27.85 ms and NVFP4 took 51.87 ms. The original compiled path loses one validation-correct example, so compilation is not assumed numerically identical. NVFP4 is retained as a measured negative result, not used for submission. It changes storage bytes and arithmetic precision, not logical parameters or dense mathematical FLOPs.

Branches `feature/01-baseline` through `feature/15-accuracy-integration` preserve the initial sequence and accuracy-first extension. Follow-up branches are `feature/16-garment-discrimination`, `feature/17-nvfp4-benchmark`, `feature/18-spatial-readout`, `feature/19-confidence-cascade`, and `feature/20-followup-integration`. Each branch remains available after merging into `main`.

The validation-only follow-up on `feature/21-image-processing` tested mild gamma, intensity-gain and half-pixel centering changes. None beat the original cascade's 95.35% validation accuracy. `feature/22-awq` tested real native torchao AWQ W4A16: validation accuracy was 95.1833%, versus 95.2333% for plain INT4 and 95.35% for original/channel-BF16 controls. On the same 1,024 validation inputs, eager median latency was 880.64 ms for AWQ, 853.94 ms for plain INT4, 249.92 ms for original and 231.89 ms for channel-BF16. This backend's padding to input multiples of 1,024 is costly for small Mixer matrices. AWQ stored tensors occupy 7,331,088 bytes versus 8,830,224 original and 5,422,224 channel-BF16 bytes, including layout/scaling overhead. No candidate was adopted; test accuracy remains the previously measured **94.42%**. These are shared-device, backend-specific results. Integration and the executed notebook are retained on `feature/23-processing-awq-integration`.

Reproduce these optional experiments with `python3 image_processing_experiment.py` and `.venv/bin/python benchmark_awq.py`; the latter uses the same pinned torchao environment described below. Evidence: `results/image_processing.json` and `results/awq_benchmark.json`. AWQ calibration uses training images only, and both experiments hold the selected inference recipe fixed.

See [JOURNAL.md](JOURNAL.md) for dated experiment notes. Results are recorded only after a run finishes. Submission marks and bonus ranking depend on instructor assessment.

The detector-inspired follow-up uses pure MLPs: a 196-to-49-token hierarchy, fine/coarse feature fusion, a training-only auxiliary classifier, and CSP-inspired partial-channel transforms. Each of four architectures receives 120 initial plus 30 clean-refinement epochs, giving eight runs and 24 checkpoints. Best single-view validation accuracy is 93.50% for the coarse-only control, 93.30% with feature fusion, 93.20% with auxiliary supervision and 93.3333% with partial channels. Partial-channel fusion uses 550,393 parameters and 47,042,880 MACs, versus 827,305 parameters and 68,718,912 MACs for full-channel fusion. It is smaller, but does not outperform the existing deployment. These are detector-inspired design ideas, not YOLO or PGI reproductions.

The 77 bounded validation comparisons retained the original cascade (5,721 correct); the best new combination had 5,714 correct. No combined auxiliary+CSP trial was added because auxiliary supervision did not improve its matched control. The final 94.42% test recipe is unchanged. Evidence: [architecture comparison](results/yolo_summary.json), [all candidates](results/yolo_candidates.csv), [protocol](results/yolo_protocol.json). Run `python3 check_pyramid.py`, then `python3 run_yolo.py` and `python3 select_yolo.py`; `run_yolo.py --retrain` explicitly replaces recorded training outputs. Major stages remain on `feature/24-multiscale-mlp`, `feature/25-auxiliary-supervision`, `feature/26-partial-channel-mlp`, `feature/27-yolo-evaluation`, and `feature/28-yolo-integration`.

## Run an experiment

Use Python 3.12 and the dependencies in `requirements.txt`. The recorded environment uses PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128 on an RTX 5090. CPU is supported, but the longer Mixer experiments are substantially slower and BF16 autocast is disabled there.

```bash
python3 check_models.py
python3 sharpness.py
python3 baseline.py
python3 train.py configs/adamw.json
python3 train.py configs/mixer.json
python3 train.py configs/mixer_muon.json
python3 check_garment.py
python3 check_cascade.py
python3 benchmark_cascade.py
```

The optional real FP4 comparison uses an isolated environment with the existing PyTorch and `torchao==0.16.0`: `python3 -m venv --system-site-packages .venv`, `.venv/bin/python -m pip install --no-deps torchao==0.16.0`, then `.venv/bin/python benchmark_nvfp4.py --compile --recipe results/spatial_recipe.json`. The recipe argument reproduces the ungated model used by the recorded NVFP4 comparison. CUDA FP4 hardware is required; these packages are not needed for default notebook evaluation.

Run from the repository root. Fashion-MNIST downloads into `data/`. Every run writes its measured epoch history to `results/`, saves the validation-selected checkpoint, and appends its result to the journal. Checkpoints and dataset files are ignored by Git. Rerunning a configuration replaces its local result/checkpoint files; use a separate checkout or directory when preserving the recorded run.

`baseline.py` normalizes and flattens inputs outside its model; `train.Model` performs normalization internally. Do not normalize inputs twice when evaluating saved checkpoints. Cross-entropy takes raw logits; softmax is used for probabilities at inference, avoiding redundant softmax during training.

See [SOURCES.md](SOURCES.md) for source links and complexity-counting conventions.
