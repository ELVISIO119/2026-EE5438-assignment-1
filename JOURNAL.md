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

## 2026-09-18T08:05:16.447061+00:00 — residual_muon

Hypothesis: Change only hidden matrix optimizer to native Muon with RMS-matched updates versus residual_swiglu; input/head/norm/bias stay on AdamW.

Measured validation accuracy: 91.43%; checkpoint epoch 40; 994,056 parameters; 988,928 dense MACs/image; 42.6s training/validation wall time. Configuration and every epoch: `results/residual_muon.json`. Test set not evaluated.

## 2026-09-18T08:06:42.235904+00:00 — residual_sam

Hypothesis: SAM with AdamW base optimizer versus residual_swiglu. Same epochs but twice the gradient evaluations; independent dropout masks at the two evaluations.

Measured validation accuracy: 90.62%; checkpoint epoch 36; 994,056 parameters; 988,928 dense MACs/image; 52.6s training/validation wall time. Configuration and every epoch: `results/residual_sam.json`. Test set not evaluated.

## 2026-09-18T08:07:38.203396+00:00 — residual_asam

Hypothesis: ASAM with weight-scaled neighborhood eta 0.01 versus ordinary SAM and residual_swiglu. Rho differs by method; this is a method-setting comparison, not an isolated radius ablation.

Measured validation accuracy: 90.50%; checkpoint epoch 37; 994,056 parameters; 988,928 dense MACs/image; 53.6s training/validation wall time. Configuration and every epoch: `results/residual_asam.json`. Test set not evaluated.

## Interpretation before completing the Mixer trial

The fixed sigmoid/SGD baseline is substantially undertrained at learning rate 0.001 and 50 epochs. Its low score is a reference for this specified recipe, not evidence that SGD or sigmoid cannot perform well. AdamW improves the same architecture to 89.78%; the residual GELU configuration reaches 90.97%, but changes architecture, schedule and regularization together, so those gains cannot be assigned to residual connections alone.

Within the matched residual trials, SiLU and BatchNorm do not outperform GELU/LayerNorm. Geometry changes validation accuracy only slightly (90.97% to 90.98%); adding Mixup reduces it to 90.50%. Replacing the feedforward with approximately parameter-matched SwiGLU improves the Mixup configuration to 91.10%. This does not demonstrate that Mixup is necessary for SwiGLU.

Muon/AdamW reaches 91.43%, versus 91.10% for the matched AdamW configuration, but takes 42.6 rather than 21.5 seconds here. This does not reproduce a large-language-model compute-efficiency claim. SAM (90.62%) and ASAM (90.50%) cost two gradient evaluations per batch without an accuracy gain at the tested settings. Keep these negative results; do not include these optimizers in the final recipe merely because they are advanced methods.

The Mixer trial uses 120 instead of 40 epochs and adds label smoothing while removing Mixup. It tests a complete architecture/training recipe, not a causal architecture-only effect. Its 478,640 parameters are fewer than the residual model's, but its 29,003,008 dense MACs per image are substantially higher because weights are reused across tokens. Report both costs.

## 2026-09-18T08:12:27.855407+00:00 — mixer_adamw

Hypothesis: Pure dense token/channel mixing may exploit image layout without convolution or attention. This longer-budget architecture trial is not a one-variable optimizer ablation; no Mixup, mild geometry and label smoothing 0.05.

Measured validation accuracy: 93.68%; checkpoint epoch 86; 478,640 parameters; 29,003,008 dense MACs/image; 194.5s training/validation wall time. Configuration and every epoch: `results/mixer_adamw.json`. Test set not evaluated.

## 2026-09-18T08:18:29.316851+00:00 — mixer_muon

Hypothesis: Change only hidden-matrix optimization relative to mixer_adamw, retaining matched epochs, seed, architecture and regularization. Patch embedding, head, normalization and bias stay on AdamW. Measure actual time rather than transferring an LLM efficiency claim.

Measured validation accuracy: 93.88%; checkpoint epoch 82; 478,640 parameters; 29,003,008 dense MACs/image; 346.4s training/validation wall time. Configuration and every epoch: `results/mixer_muon.json`. Test set not evaluated.

## 2026-09-18T08:20:18.694771+00:00 — mixer_average

Hypothesis: Fine-tune the validation-selected AdamW Mixer on clean training images, comparing ordinary weights, batchwise EMA and late-epoch SWA on exactly the same optimization trajectory. Clean fine-tuning itself changes the recipe; averaging effects use the within-run control.

Measured validation accuracy: 93.88%; checkpoint epoch 4; 478,640 parameters; 29,003,008 dense MACs/image; 50.2s training/validation wall time. Configuration and every epoch: `results/mixer_average.json`. Test set not evaluated.

### 2026-09-18T08:20:18.694771+00:00 - mixer_average EMA

Validation accuracy 94.02% at epoch 4. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/mixer_average_ema.json`.

### 2026-09-18T08:20:18.694771+00:00 - mixer_average SWA

Validation accuracy 93.67% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/mixer_average_swa.json`.

## 2026-09-18T08:22:42.695836+00:00 — student_control

Hypothesis: Small dense Mixer with hard labels provides a same-initialization control for distillation; compare accuracy and inference MACs with the larger teacher. The 80-epoch student budget differs from the 120-epoch teacher.

Measured validation accuracy: 92.32%; checkpoint epoch 70; 90,525 parameters; 4,867,712 dense MACs/image; 71.8s training/validation wall time. Configuration and every epoch: `results/student_control.json`. Test set not evaluated.

## 2026-09-18T08:24:28.351214+00:00 — student_distill

Hypothesis: Compare the same small student's control with 50% smoothed-label CE plus 50% teacher KL at temperature 3 (including T squared scaling). The frozen teacher is an in-scope MLP trained only on the training partition; teacher inference adds training cost, not student inference cost.

Measured validation accuracy: 92.32%; checkpoint epoch 77; 90,525 parameters; 4,867,712 dense MACs/image; 103.2s training/validation wall time. Configuration and every epoch: `results/student_distill.json`. Test set not evaluated.

## 2026-09-18T08:24:43.330101+00:00 - mixer_pruned_75 before fine-tuning

Keep 75% of channel neurons: validation 91.47%; 379,952 parameters; 24,186,112 MACs/image. Weights, not validation labels, determine neuron ranking. Evidence: `results/mixer_pruned_75_initial.json`.

## 2026-09-18T08:25:20.537611+00:00 — mixer_pruned_75

Hypothesis: Physically retain 75% of channel-MLP neurons ranked by incoming/outgoing weight norms, then clean fine-tune. Compare pre- and post-fine-tuning validation accuracy and real matrix dimensions. Source: mixer_adamw.

Measured validation accuracy: 93.70%; checkpoint epoch 4; 379,952 parameters; 24,186,112 dense MACs/image; 37.0s training/validation wall time. Configuration and every epoch: `results/mixer_pruned_75.json`. Test set not evaluated.

## 2026-09-18T08:25:20.583967+00:00 - mixer_pruned_50 before fine-tuning

Keep 50% of channel neurons: validation 81.23%; 281,264 parameters; 19,369,216 MACs/image. Weights, not validation labels, determine neuron ranking. Evidence: `results/mixer_pruned_50_initial.json`.

## 2026-09-18T08:25:59.042585+00:00 — mixer_pruned_50

Hypothesis: Physically retain 50% of channel-MLP neurons ranked by incoming/outgoing weight norms, then clean fine-tune. Compare pre- and post-fine-tuning validation accuracy and real matrix dimensions. Source: mixer_adamw.

Measured validation accuracy: 93.47%; checkpoint epoch 12; 281,264 parameters; 19,369,216 dense MACs/image; 38.3s training/validation wall time. Configuration and every epoch: `results/mixer_pruned_50.json`. Test set not evaluated.

## 2026-09-18T08:26:24.379846+00:00 — regularization_none

Hypothesis: Complete a 2x2 dropout/decoupled-weight-decay comparison around residual_gelu. This cell has neither dropout nor weight decay; data, architecture, initialization, batch size and schedule are held fixed.

Measured validation accuracy: 90.72%; checkpoint epoch 33; 994,314 parameters; 989,696 dense MACs/image; 14.1s training/validation wall time. Configuration and every epoch: `results/regularization_none.json`. Test set not evaluated.

## 2026-09-18T08:26:41.608567+00:00 — regularization_dropout

Hypothesis: Enable only dropout p=0.1 versus regularization_none, holding weight decay at zero. Dropout necessarily consumes random draws; this is a fixed-seed comparison, not identical stochastic trajectories.

Measured validation accuracy: 90.98%; checkpoint epoch 24; 994,314 parameters; 989,696 dense MACs/image; 14.9s training/validation wall time. Configuration and every epoch: `results/regularization_dropout.json`. Test set not evaluated.

## 2026-09-18T08:26:58.204215+00:00 — regularization_decay

Hypothesis: Enable only decoupled weight decay 0.01 versus regularization_none. Compare with dropout-only and residual_gelu (both enabled) to describe regularization without conflating it with architecture changes.

Measured validation accuracy: 90.68%; checkpoint epoch 17; 994,314 parameters; 989,696 dense MACs/image; 14.2s training/validation wall time. Configuration and every epoch: `results/regularization_decay.json`. Test set not evaluated.

## 2026-09-18T08:28:40.312871+00:00 - Freeze final recipe

Selected `mixer_average:4 + mixer_average_swa:4 + mixer_muon:4` from 31 validation-only candidates: 5680/6000 correct (94.67%). Total stored parameters 1,435,920; dense MACs/image 348,036,096, including all members and views. Checkpoint hashes and the selection rule are in `results/final_recipe.json`. No test labels were loaded by selection.
