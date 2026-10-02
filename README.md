# EE5438 Assignment 1 — pure MLP Fashion-MNIST

Student: Cai Haochen (`58561440`)

Branch: `feature/49-latent-search-integration`

Repository: <https://github.com/ELVISIO119/2026-EE5438-assignment-1>

## How the 94.55% model was trained

The model was built in one measured path. All experiments use the official Fashion-MNIST training set, seed `58561440`, and a fixed stratified split of 54,000 fitting images and 6,000 validation images.

1. **Baseline.** A `784 -> 128 -> 64 -> 10` sigmoid MLP was trained with SGD for 50 epochs.
2. **Better optimisation.** AdamW was tested on the same small MLP, then residual GELU/SwiGLU MLPs were tested with LayerNorm, dropout, weight decay, warmup/cosine decay and mild geometry augmentation.
3. **MLP-Mixer direction.** Patch embedding plus token/channel `Linear` mixing gave a stronger pure-MLP model. AdamW and Muon+AdamW were compared; Muon was used only for hidden matrices.
4. **Accuracy checkpoints.** A wider Mixer, a finer-patch Mixer and a structured-pruned Mixer were clean-fine-tuned and evaluated with EMA/SWA. The best complementary checkpoints were retained.
5. **Final inference recipe.** Five fixed checkpoints were combined: `legacy_clean`, `legacy_muon`, `accuracy_wide_clean_swa`, `accuracy_p2_clean_swa` and `mixer_pruned_75`. A seeded 4,096-candidate search tuned only confidence thresholds, ensemble weights and temperatures on validation data. It did not change network weights.

No CNN, Transformer, attention, external training images or pretrained weights are used. The final network components are feed-forward `Linear` layers.

## Main training results

| Stage | Validation accuracy | Parameters | Dense MACs/image |
|---|---:|---:|---:|
| Sigmoid + SGD baseline | 55.22% | 109,386 | 109,184 |
| Same MLP + AdamW | 89.78% | 109,386 | 109,184 |
| Residual GELU | 90.97% | 994,314 | 989,696 |
| Residual SwiGLU | 91.10% | 994,056 | 988,928 |
| Residual Muon + AdamW | 91.43% | 994,056 | 988,928 |
| Mixer + AdamW | 93.68% | 478,640 | 29,003,008 |
| Mixer + Muon + AdamW | 93.88% | 478,640 | 29,003,008 |
| Wide Mixer, clean fine-tuning + SWA | 94.03% | 1,297,746 | 77,222,784 |
| Fine-patch Mixer, clean fine-tuning + SWA | 93.85% | 529,858 | 72,329,664 |
| Structured-pruned Mixer + fine-tuning | 93.70% | 379,952 | 24,186,112 |

The table records validation checkpoints. The final 94.55% number comes from the frozen five-checkpoint ensemble and its validation-selected inference route, not from claiming that one individual checkpoint reaches 94.55%.

## Final benchmark

| Split | Correct | Accuracy | Macro precision | Macro recall | Macro F1 | Shirt F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation (6,000) | 5,720 | 95.33% | — | — | 0.9528 | 0.8598 |
| Test (10,000) | 9,455 | **94.55%** | 0.9453 | 0.9455 | 0.9453 | 0.8316 |

Stored parameters are 3,164,836. Average test cost is 301,330,871 MACs/image; worst-case routed cost is 3,319,207,680 MACs/image.

## Reproduce the submitted result

The notebook is the main pipeline. It contains the executable code and the five required checkpoints in an embedded bundle, so a fresh clone does not need untracked `.pt` files.

```bash
git clone https://github.com/ELVISIO119/2026-EE5438-assignment-1.git
cd 2026-EE5438-assignment-1
git checkout feature/49-latent-search-integration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter
jupyter nbconvert --execute --to notebook --output executed.ipynb \
  submission/Assign01_Cai_Haochen_58561440.ipynb
```

Run All downloads Fashion-MNIST, recreates the fixed split, checks the MLP implementations, evaluates the frozen route and prints the test report. It does not retrain by default. The first setup cell contains `RETRAIN_ALL = False`; changing it to `True` reruns the long training experiments and requires a new validation freeze before using new weights.

## Fixed data protocol

| Item | Value |
|---|---|
| Dataset | Official Fashion-MNIST |
| Input | `1 x 28 x 28`, scaled to `[0, 1]` |
| Seed | `58561440` |
| Fit/validation split | 54,000 / 6,000, stratified |
| Validation balance | 600 images per class |
| Normalisation | Mean and standard deviation fitted on fitting images only |
| Test use | Read after the route was frozen |

## Optional source checks

These commands are not needed for a fresh notebook reproduction. They are useful when the matching checkpoints are already present in `results/`:

```bash
python3 check_models.py
python3 latent_search.py --check
python3 -c "from refine_cascade import evaluate; evaluate('latent')"
```

The search uses 16 populations of 256 candidates (4,096 total), keeps 209 validation-qualified candidates, and selects candidate 4072. It tunes inference settings only; it does not use test labels.

## Important files

| File | Purpose |
|---|---|
| `submission/Assign01_Cai_Haochen_58561440.ipynb` | Complete assignment notebook and primary reproduction entry point |
| `train.py` | Seeded split, model wrapper and training loop |
| `models.py` | Residual and Mixer-style pure-MLP architectures |
| `configs/*.json` | Training settings for the retained experiments |
| `latent_search.py` | Deterministic validation-only route search |
| `refine_cascade.py` | Route freezing and endpoint evaluation |
| `results/latent_balanced_recipe.json` | Frozen final route |
| `results/latent_test_metrics.json` | Final test metrics and confusion matrix |

The 94.55% score is a measured Fashion-MNIST benchmark; it is not a guarantee of a class ranking or an independently blinded holdout result.
