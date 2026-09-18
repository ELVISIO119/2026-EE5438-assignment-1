# Experiment journal

## Project setup

Objective: improve Fashion-MNIST classification using only MLPs while keeping a clear record of architecture, optimizer, regularization and inference costs.

First task: establish the supplied-style sigmoid/SGD baseline with seed 58561440. Reserve 6,000 examples for validation. The completed run reached **54.78% validation accuracy** at epoch 50. This is the reference to beat; the next branch changes only the optimizer.

The baseline is intentionally plain: 784 inputs → 128 sigmoid units → 64 sigmoid units → 10 logits, cross-entropy loss, SGD at 0.001, batch size 128 and 50 epochs. Training took about 5.6 seconds on the recorded CUDA environment. The result is evidence, not a test-set score.

For every feature, record its hypothesis, the exact comparison, measured outcome, limitations and the next decision. Record wall-clock timestamps when measurements finish; never backdate entries.

## 2026-09-18T07:59:08Z — Correct the baseline split

The first implementation accidentally used an unstratified split. Changed it to 600 validation images per class, added split assertions, saved indices and reran the self-contained notebook. The corrected validation result is **55.22%**, replacing the 54.78% trial as the comparison baseline. No test score was used.

## 2026-09-18T08:00:16.806060+00:00 — adamw_sigmoid

Hypothesis: Changing SGD to AdamW alone should accelerate the fixed sigmoid architecture. Same learning rate, split, seed and 50-epoch budget as baseline.

Measured validation accuracy: 89.78%; checkpoint epoch 28; 109,386 parameters; 109,184 dense MACs/image; 8.3s training/validation wall time. Configuration and every epoch: `results/adamw_sigmoid.json`. Test set not evaluated.

## 2026-09-18T08:01:07.751349+00:00 — residual_gelu

Hypothesis: Combined modern reference: residual GELU blocks, LayerNorm, dropout, weight decay and warmup/cosine scheduling. This is not a one-factor ablation against the sigmoid model.

Measured validation accuracy: 90.97%; checkpoint epoch 34; 994,314 parameters; 989,696 dense MACs/image; 13.6s training/validation wall time. Configuration and every epoch: `results/residual_gelu.json`. Test set not evaluated.

## 2026-09-18T08:01:24.037170+00:00 — residual_silu

Hypothesis: Change GELU to SiLU only relative to residual_gelu.

Measured validation accuracy: 90.80%; checkpoint epoch 30; 994,314 parameters; 989,696 dense MACs/image; 13.9s training/validation wall time. Configuration and every epoch: `results/residual_silu.json`. Test set not evaluated.

## 2026-09-18T08:01:41.269091+00:00 — residual_bn

Hypothesis: Replace block LayerNorm with BatchNorm only relative to residual_gelu; keep final LayerNorm.

Measured validation accuracy: 90.85%; checkpoint epoch 36; 994,314 parameters; 989,696 dense MACs/image; 14.9s training/validation wall time. Configuration and every epoch: `results/residual_bn.json`. Test set not evaluated.

## 2026-09-18T08:02:30.891210+00:00 — residual_geometry

Hypothesis: Add only mild horizontal flips, rotations, translations and scaling to residual_gelu.

Measured validation accuracy: 90.98%; checkpoint epoch 39; 994,314 parameters; 989,696 dense MACs/image; 18.4s training/validation wall time. Configuration and every epoch: `results/residual_geometry.json`. Test set not evaluated.

## 2026-09-18T08:02:52.604011+00:00 — residual_mixup

Hypothesis: Add Mixup alpha 0.2 only relative to residual_geometry. Training accuracy against original hard labels is descriptive only, not equivalent to clean accuracy.

Measured validation accuracy: 90.50%; checkpoint epoch 37; 994,314 parameters; 989,696 dense MACs/image; 19.4s training/validation wall time. Configuration and every epoch: `results/residual_mixup.json`. Test set not evaluated.

## 2026-09-18T08:03:41.687083+00:00 — residual_swiglu

Hypothesis: Replace GELU feedforward with SwiGLU at approximately equal parameter count relative to residual_mixup. Gate and value weights remain separate.

Measured validation accuracy: 91.10%; checkpoint epoch 37; 994,056 parameters; 988,928 dense MACs/image; 21.5s training/validation wall time. Configuration and every epoch: `results/residual_swiglu.json`. Test set not evaluated.
