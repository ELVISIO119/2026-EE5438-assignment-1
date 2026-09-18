# Experiment journal

## 2026-09-19 — Recover the earlier efficient baseline (feature/36-legacy-baseline)

Imported the earlier same-student `mixer_p4_clean_finetune` and `mixer_p4_muon` checkpoints as `legacy_clean` and `legacy_muon`. The source folder was read-only. Only state-dictionary names changed; all 6,000 validation logits matched the original model class exactly. Training split, seed and normalization match the current project. Each model has 478,640 parameters and 29,003,008 dense MACs per view. Their equal-weight 10-view ensemble reproduces 5,695/6,000 validation correct (94.9167%), 957,280 parameters and 580,060,160 MACs/image. Its 94.16% test score is a historical record, not a new test evaluation or an unobserved benchmark.

Source hashes, original configurations and import checks are in `results/legacy_import.json` and the two checkpoint records. Imported models are earlier training on the same assignment data, not external pretraining. They remain separately named so provenance is not rewritten into the newer experiment history.

## 2026-09-19 — MoE and loss integration (feature/35-moe-loss-integration)

Re-evaluated all 115 saved checkpoints and the unchanged frozen cascade successfully. The portable notebook executed with 13 figures, adding expert occupancy/latency and class-recall comparisons. The two-file ZIP passed execution, archive, metric and credential-history checks. The prior archive is retained as `archives/Assign01_Cai_Haochen_58561440_before_moe_loss.zip`. The final recipe remains byte-for-byte equivalent as parsed JSON to both experiment reference snapshots, including its original freeze timestamp. Section A remains a typed study guide that must be replaced by the student's genuine handwritten scan; no Canvas upload was performed. Shared-trunk experts were tested; independently bootstrap-trained bagging and garment-only experts were not represented as completed experiments.

## 2026-09-19 — Loss-follow-up outcome (local date; raw JSON timestamps are UTC)

Selected raw-validation checkpoints: CE `features_gray` 5,642/6,000, gamma-2 focal `loss_focal2_ema` 5,644, weighted CE `loss_weighted_ema` 5,641. Shirt correct counts are 485, 485 and 483 out of 600. Focal raises precision from 0.842014 to 0.847902 and F1 from 0.824830 to 0.827645 without increasing recall. Its fixed-half counts change from [2,808, 2,834] to [2,806, 2,838], failing the prospective advancement rule; weighted CE also fails. Neither is stacked with local augmentation or garment-only post-training. This is a bounded decision under reused validation, not statistical rejection of the methods.

All 58 deployment comparisons retain the original 5,721-correct cascade. No new candidate test evaluation was performed; final weights and the original freeze timestamp remain unchanged. Evidence: `results/loss_experiment.json`, `results/loss_candidates.csv`, `results/loss_recipe.json`.

## 2026-09-18 — Focal and weighted-CE follow-up design (feature/34-garment-loss-followup)

The user proposed focal gamma 2, class-weighted CE, targeted augmentation and garment-only post-training. The earlier gamma-1 and conditional-garment objectives already exist; this follow-up isolates gamma 2 and mild weight 1.5 for T-shirt/Pullover/Coat/Shirt. Fashion-MNIST is balanced, so these weights express difficulty preference, not imbalance correction. Weighted CE divides by the sum of sample weights, matching native PyTorch. Self-checks compare both losses with independent expressions and verify gradients.

Both new configurations match `features_gray`: wide SWA initialization, 30 clean epochs, same optimizer, schedule, seed, architecture and averaging. The existing completed CE run is reused as the exact matching control. All training classes remain present. Only if a selected loss improves overall correct count while preserving Shirt recall/F1 and both fixed development-half counts will local augmentation be stacked. No aggressive upper-half crop or garment-only replacement of the shared classifier is assumed safe: these can discard discriminative information or cause forgetting. Three selected families enter the existing 58-candidate validation deployment grid; no test-driven selection.

## 2026-09-18 — Sparse-expert outcome

The selected three-expert EMA checkpoint achieves 5,641/6,000 validation correct (94.0167%) versus 5,642 (94.0333%) for the approximately parameter-matched one-expert EMA control. Route counts are [2,308, 2,330, 1,362]; every expert is used. Dense soft inference on the selected sparse checkpoint also yields 5,641 correct. The SWA checkpoint loses nine correct predictions when switching dense to hard routing, documenting a possible train/inference mismatch.

Stored parameters are 1,742,421 (MoE) versus 1,741,651 (control). Top-1 MoE uses 77,370,816 dense MACs per image, versus 77,665,728 for soft all-expert inference. Most computation remains in the shared trunk. For 1,024 images, measured eager forward-only medians are 7.8068 ms (parent), 9.2951 ms (single-expert control), 10.9350 ms (top-1 MoE), and 8.6472 ms (dense soft MoE). Sparse dispatch is slower on this small model/backend, despite executing fewer expert MACs. These are shared-device measurements. All 39 deployment candidates retain the original 5,721-correct cascade; final weights and the original freeze timestamp remain unchanged.

## 2026-09-18 — Shared-trunk sparse expert design (feature/33-moe-routing)

Test three width-384 residual MLP experts after the wide Mixer's pooled features, with a learned Linear router and original classifier. Train a soft weighted feature mixture and deploy top-1 dispatch to genuinely skip unselected expert rows. A single width-1152 residual MLP is the approximately total-parameter-matched control. Both start from the same wide SWA parent and receive 30 clean epochs with the existing AdamW schedule. Expert output layers start at zero; extra initialization preserves the parent's RNG stream. Expert branches use no dropout, keeping the shared-trunk dropout and shuffle stream matched. All layers, including the trunk, are fine-tuned. This changes readout capacity rather than sparsifying expensive trunk blocks.

Ordinary/EMA/SWA checkpoints are selected by actual top-1 validation accuracy then NLL. Diagnose the train/eval mismatch using dense soft inference on those same checkpoints. There is no balancing penalty; record route occupancy instead of assuming balanced experts. This is shared-trunk MoE-inspired training, not independent bootstrap bagging, a class-specialist experiment, or token generation/speculative decoding. Six checkpoint evaluations, two selected-family deployment grids (39 candidates), and fixed 1,024-image latency comparisons use validation data only. The existing champion remains a candidate with both development-half counts and Shirt F1 guarded. Report full stored parameters, sparse and dense forward MACs, and three-warmup/seven-repeat batch-128 eager timings on the shared GPU. Expert selection/memory overhead is not represented by dense Linear MACs.

Commands: `python3 moe_experiment.py --check`, then `python3 moe_experiment.py`. Existing matching completed training outputs are reused; each newly trained configuration has a 1,800-second hard timeout. The experiment is a single-seed short fine-tuning test, not a statistical superiority claim.

## 2026-09-18 — Image-feature integration (feature/32-image-feature-integration)

Re-evaluated all 103 saved checkpoints and the unchanged frozen cascade successfully. The portable notebook executed with 11 figures, including garment confusion matrices and grayscale/gradient/contrast views of three difficult validation Shirts. The regenerated two-file ZIP passed execution, archive, metric and credential-history checks. The pre-experiment archive remains in `archives/Assign01_Cai_Haochen_58561440_before_image_features.zip`. The Section A PDF still needs replacement with the student's genuine handwritten scan; no Canvas upload was performed.

## 2026-09-18 — Image-feature outcome

Completed all three matched runs and nine checkpoint variants. Selected grayscale and gradient checkpoints each classify 5,642/6,000 validation images correctly; gradients plus local contrast give 5,640. Shirt correct counts are 485, 484 and 483 of 600. Gradient Shirt F1 rises from 0.824830 to 0.825939 because false positives decline, while recall falls. This provides no support for promoting these fixed features under the tested warm-start and 30-epoch budget.

The 58 bounded standalone/addition/replacement deployment candidates retain the original 5,721-correct cascade; the strongest alternative reaches 5,718. Final weights, deployment recipe and original freeze timestamp remain unchanged. No candidate test evaluation was performed. Detailed confusion matrices and all candidate results are retained in `results/features_details.json` and `results/features_candidates.csv`.

## 2026-09-18 — Paired fixed-image-feature design (feature/31-image-features)

The user's new hypothesis concerns explicit garment outlines and local intensity structure, rather than further model capacity. Three configurations were committed before running: retained grayscale, grayscale plus signed horizontal/vertical central differences, and those channels plus grayscale minus its 3x3 local mean. Replicate padding avoids constant-background boundary artifacts. Features are generated inside the input path after geometry, so every training/inference view uses the same processing. No labels or fitted feature statistics enter the transform.

All trials initialize from `accuracy_wide_clean_swa` and use 30 clean fine-tuning epochs with the same AdamW, 5e-5 learning rate, 0.01 decay, warmup/cosine schedule, dropout and seed. Extra input columns start at zero. Model construction preserves the control's initialization RNG consumption; the check verifies unchanged FP32 initial logits, gradient directions, constant backgrounds and finite gradients. Shape-dependent BF16 rounding is possible. The original/EMA/SWA checkpoint rule and bounded deployment search reuse existing helpers. Dense MACs include expanded embedding matrices; fixed-filter arithmetic is separately disclosed. Test data are excluded from new fitting and selection.

## 2026-09-18 — PIL integration (feature/30-pil-integration)

All 94 saved checkpoints and the original cascade passed re-evaluation. The portable notebook executed with nine figures, and the regenerated two-file ZIP passed submission checks. The original model and freeze timestamp were preserved. The Section A PDF remains a typed study guide, requiring the student's genuine handwritten scan before submission.

## 2026-09-18 — PIL-inspired ridge readouts (feature/29-pil-ridge-heads)

Froze all feature-extractor weights in the three selected ensemble members. On 54,000 training images, extracted the BF16 features actually entering each classifier and solved centered ridge regression to ten-class one-hot targets using CPU float64 thin SVD. The intercept is unpenalized; lambda zero is a thresholded pseudoinverse. Tested eight fixed lambdas (0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1), selected by actual BF16 validation correct count then NLL. A rank-deficient numerical self-check agrees with augmented least squares; cached classifier evaluation is bitwise identical to the full model, and every non-head checkpoint tensor remains exactly unchanged.

Single-view validation correct counts: wide 5,642 -> 5,651 (94.1833%, lambda 1e-4), fine-patch 5,631 -> 5,634 (93.90%, lambda 0.1), pruned 5,622 -> 5,622 (93.70%, lambda 0.01). Wide-model Shirt F1 slightly decreased despite the overall gain. These are small single-split observations, not significant generalization claims. Each head keeps the same dimensions, parameters and dense MACs. The feature extractors still required their original gradient-based training; this is not full-network PIL reproduction or training from scratch in one step.

Calibrated each refitted member on validation after its original TTA views using temperatures [0.05, 0.1, 0.2, 0.5, 1, 2]; all selected 0.1. Compared the original cascade, three standalone refits, nine additions and three combinations replacing non-gate members (16 deployment candidates). The original gate remains unchanged and shares the original wide member. Added models are conservatively counted/executed as complete extra backbones. Original cascade: 5,721 correct; best non-reference candidate: 5,710. Original recipe and 94.42% historical test outcome are retained, with no test-driven selection. Evidence: `results/pil_experiment.json`, `results/pil_candidates.csv`, `results/pil_recipe.json`; exact timings and timestamps are in JSON.

## 2026-09-18 — Detector-inspired experiment integration

The updated portable notebook includes all four architecture comparisons, training/refinement configurations, the bounded validation selector, and a new accuracy-versus-MAC figure. All 91 saved checkpoints and the original cascade were re-evaluated successfully; the final recipe is unchanged. Default Run All executed successfully with eight figures, and the two-file ZIP passed the submission/credential checks. The pre-experiment ZIP remains in `archives/Assign01_Cai_Haochen_58561440_before_yolo.zip`. The typed Section A PDF still requires replacement with the student's genuine handwritten scan.

## 2026-09-18T11:57:52.425576+00:00 — Detector-inspired MLP selection

Completed four paired architecture recipes, each with 120 initial epochs plus 30 clean-refinement epochs (eight runs, 24 ordinary/EMA/SWA checkpoints). Best single-view validation results: coarse-only hierarchy 93.50%, fine/coarse feature fusion 93.30%, auxiliary-supervised fusion 93.20%, partial-channel fusion 93.3333%. Partial-channel processing reduced deployment parameters from 827,305 to 550,393 and MACs from 68,718,912 to 47,042,880 relative to full-channel fusion. This small observed accuracy difference does not establish a general advantage; Shirt F1 was lower. Auxiliary supervision added 970 training-only parameters, excluded from deployment checkpoints. Clean refinement recreated that auxiliary head because only deployment state was retained.

The fixed validation comparison evaluated 77 standalone/addition/replacement recipes, with one best raw-validation checkpoint per architecture, views 1/10/30 and the original confidence gate held fixed. The original 5,721/6,000-correct cascade won; the strongest non-reference candidate achieved 5,714. Neither feature fusion nor auxiliary supervision justified further stacking, so no combined auxiliary+CSP trial was introduced. The final recipe and historical 94.42% test score remain unchanged. Selection never loaded test labels. Evidence: `results/yolo_summary.json`, `results/yolo_candidates.csv`, `results/yolo_recipe.json`. The separate original recipe remains in `results/final_recipe.json` with its original freeze timestamp.

## 2026-09-18 — Processing/AWQ integration (feature/23-processing-awq-integration)

Included both complete negative-result comparisons and their executable scripts in the portable notebook. Default Run All still evaluates the unchanged frozen submission weights; it does not require torchao or recalibrate quantized weights. The freshly executed notebook reproduces 94.42% test accuracy and passes the submission check (all code cells executed, seven figures, valid two-file ZIP, no credential patterns). This is a consistency rerun, not a new test-driven model-selection round. The previous ZIP is preserved locally in `archives/Assign01_Cai_Haochen_58561440_before_awq.zip`. Section A remains a typed study guide requiring replacement with the student's genuine handwritten scan.

## 2026-09-18 — Native AWQ deployment comparison (feature/22-awq)

Calibrated native torchao 0.16.0 AWQ on 100 training images, ten per class, with 20 activation-aware scale candidates and group-32 tile-packed INT4 channel weights. Other layers retain their original precision; activations use BF16. The observer initially captured pre-autocast FP32 LayerNorm output, causing a dtype mismatch in the native offline scale search. Casting the stored observations to the actual BF16 GEMM input dtype fixed calibration without changing the library or using validation images.

On the complete 6,000-image validation split, original and channel-BF16 controls both obtain 5,721 correct (95.35%); plain INT4 obtains 5,714 (95.2333%); AWQ obtains 5,711 (95.1833%). Median eager cascade time on the same 1,024 validation images: original 249.925 ms, channel-BF16 231.895 ms, plain INT4 853.942 ms, AWQ 880.644 ms. Each has three warmups and seven synchronized measurements, excluding loading/calibration. These are shared-device measurements and do not compare compiled kernels or other AWQ backends.

Actual stored tensor bytes, including quantization padding and metadata: original 8,830,224; channel-BF16 5,422,224; AWQ 7,331,088. This INT4 backend pads input widths to multiples of 1,024; tested channel widths are only 96–384. The profiler confirms `aten::_weight_int4pack_mm` and the native INT4 tinygemm kernel. This is a real packed-weight execution result, not fake quantization, but it gives no deployment advantage here. Logical parameters and mathematical dense FLOPs remain unchanged. No new test evaluation or recipe selection followed these negative results. Evidence: `results/awq_benchmark.json`.

## 2026-09-18 — Input processing follow-up (feature/21-image-processing)

Compared six prespecified pipelines using the same frozen cascade and all 6,000 validation images. Original: 5,721 correct (95.35%); gamma 0.95: 5,713; gamma 1.05: 5,716; intensity gain 0.95: 5,702; intensity gain 1.05: 5,689; center-of-intensity alignment bounded to half a pixel: 5,631. Neither validation accuracy nor Shirt F1 improved. All transformations precede the existing geometric TTA; the intensity-gain trials are multiplicative brightness/contrast adjustments with a fixed zero background. No candidate was adopted and no test labels were accessed. Evidence and actual timestamps: `results/image_processing.json`. The executable includes a direction/blank-image check for the alignment operation.

## Accuracy-first extension — objective set before new experiments

The final objective is now validation accuracy without a parameter or MAC budget. Retain the existing 54,000/6,000 stratified split and assignment seed. Add a finer-patch Mixer, a wider/deeper Mixer, augmented refinement of the Muon-trained Mixer, and a larger flat-input residual MLP. Compare ordinary, EMA and SWA checkpoints. Expand deterministic inference to 10 and 18 views, including flips and both image axes. Select ensembles on validation data, tie-breaking by validation negative log-likelihood, with no compute penalty.

The public benchmark has already been evaluated in this project. This is an exploratory continuation, not a newly blinded test. Existing test scores are historical context only; new candidate fitting, selection and stopping decisions use training/validation evidence. Freeze the new recipe before its test evaluation, and do not use that result to revise it. The previous delivery remains recoverable in Git and in a local archive.

### Inference-view check while new models train

Using the fixed `mixer_muon` checkpoint, 4 views yield 5,645/6,000 validation correct (94.08%), 10 views yield 5,663 (94.38%), and 18 views yield 5,661 (94.35%). Thus, the expanded view search has a measured benefit, but more views are not monotonically better. The final search will compare all predefined view settings rather than automatically using the most expensive one. A synthetic impulse check confirms that the 10 and 18 views are distinct zero-padded translations/flips, and a complementary-error check validates convex ensemble weighting.

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

## Initial frozen evaluation record

The validation-only search froze `mixer_average:4 + mixer_average_swa:4 + mixer_muon:4` at 2026-09-18T08:28:40Z. It selected 5,680/6,000 validation examples correctly (94.67%) under the prespecified accuracy-first rule. Each of the three checkpoints was hash-checked before reading test labels.

The one-time frozen benchmark evaluation at 2026-09-18T08:29:33Z produced 9,404/10,000 test accuracy (94.04%), macro precision 0.9402, macro recall 0.9404 and macro F1 0.9403. The fixed sigmoid/SGD reference scored 54.59% on the same test benchmark. This test result is recorded for the assignment report only; no further model, view or ensemble choice is made from it.

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

## 2026-09-18T10:04:18.879379+00:00 — accuracy_p2

Hypothesis: Accuracy-first extension: finer two-pixel patches retain spatial detail and diversify the four-pixel Mixer. Train from scratch with a longer budget and compare ordinary/EMA/SWA validation checkpoints. Cost is measured but not a selection penalty.

Measured validation accuracy: 93.50%; checkpoint epoch 145; 529,858 parameters; 72,329,664 dense MACs/image; 400.1s training/validation wall time. Configuration and every epoch: `results/accuracy_p2.json`. Test set not evaluated.

### 2026-09-18T10:04:18.879379+00:00 - accuracy_p2 EMA

Validation accuracy 93.40% at epoch 174. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_p2_ema.json`.

### 2026-09-18T10:04:18.879379+00:00 - accuracy_p2 SWA

Validation accuracy 93.50% at epoch 145. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_p2_swa.json`.

## 2026-09-18T10:11:51.463921+00:00 — accuracy_wide

Hypothesis: Accuracy-first capacity trial: wider and deeper dense Mixer, with geometry, global gradient-norm clipping at 1 and longer training. This compares complete recipes, not a one-factor architectural effect; validation only selects weights and averaging.

Measured validation accuracy: 93.47%; checkpoint epoch 101; 1,297,746 parameters; 77,222,784 dense MACs/image; 450.1s training/validation wall time. Configuration and every epoch: `results/accuracy_wide.json`. Test set not evaluated.

### 2026-09-18T10:11:51.463921+00:00 - accuracy_wide EMA

Validation accuracy 93.78% at epoch 102. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_wide_ema.json`.

### 2026-09-18T10:11:51.463921+00:00 - accuracy_wide SWA

Validation accuracy 93.30% at epoch 172. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_wide_swa.json`.

## 2026-09-18T10:13:01.790854+00:00 — accuracy_muon_refine

Hypothesis: Refine the validation-selected Muon-trained Mixer using AdamW, unsmoothed labels and continued mild geometry. EMA/SWA are measured along the same trajectory. Retain augmentation to reduce the overfitting seen in clean fine-tuning.

Measured validation accuracy: 93.85%; checkpoint epoch 2; 478,640 parameters; 29,003,008 dense MACs/image; 67.9s training/validation wall time. Configuration and every epoch: `results/accuracy_muon_refine.json`. Test set not evaluated in this run.

### 2026-09-18T10:13:01.790854+00:00 - accuracy_muon_refine EMA

Validation accuracy 93.80% at epoch 10. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_muon_refine_ema.json`.

### 2026-09-18T10:13:01.790854+00:00 - accuracy_muon_refine SWA

Validation accuracy 93.70% at epoch 34. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_muon_refine_swa.json`.

## 2026-09-18T10:14:58.859333+00:00 — accuracy_residual

Hypothesis: Train a larger flat-input residual SwiGLU MLP as a structurally different ensemble candidate. Accuracy-first selection may benefit from complementary errors, even when standalone accuracy is lower; no convolution or attention.

Measured validation accuracy: 91.93%; checkpoint epoch 115; 10,066,698 parameters; 10,046,976 dense MACs/image; 114.6s training/validation wall time. Configuration and every epoch: `results/accuracy_residual.json`. Test set not evaluated in this run.

### 2026-09-18T10:14:58.859333+00:00 - accuracy_residual EMA

Validation accuracy 91.87% at epoch 84. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_residual_ema.json`.

### 2026-09-18T10:14:58.859333+00:00 - accuracy_residual SWA

Validation accuracy 91.80% at epoch 120. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_residual_swa.json`.

## 2026-09-18T10:16:18.307306+00:00 — accuracy_p2_clean

Hypothesis: Following the measured gain from clean fine-tuning in the four-pixel Mixer, test clean unsmoothed fine-tuning of the new finer-patch model with a smaller learning rate. Compare ordinary/EMA/SWA and retain the original checkpoint as a candidate; improvement is not assumed.

Measured validation accuracy: 93.95%; checkpoint epoch 10; 529,858 parameters; 72,329,664 dense MACs/image; 63.9s training/validation wall time. Configuration and every epoch: `results/accuracy_p2_clean.json`. Test set not evaluated in this run.

### 2026-09-18T10:16:18.307306+00:00 - accuracy_p2_clean EMA

Validation accuracy 93.98% at epoch 13. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_p2_clean_ema.json`.

### 2026-09-18T10:16:18.307306+00:00 - accuracy_p2_clean SWA

Validation accuracy 93.85% at epoch 28. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_p2_clean_swa.json`.

## 2026-09-18T10:17:32.750223+00:00 — accuracy_wide_clean

Hypothesis: Test low-learning-rate clean fine-tuning of the wider Mixer after its augmented training. Compare ordinary/EMA/SWA on validation and preserve the parent checkpoint; this combined data/loss change is not an isolated causal ablation.

Measured validation accuracy: 94.10%; checkpoint epoch 20; 1,297,746 parameters; 77,222,784 dense MACs/image; 72.0s training/validation wall time. Configuration and every epoch: `results/accuracy_wide_clean.json`. Test set not evaluated in this run.

### 2026-09-18T10:17:32.750223+00:00 - accuracy_wide_clean EMA

Validation accuracy 94.08% at epoch 17. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_wide_clean_ema.json`.

### 2026-09-18T10:17:32.750223+00:00 - accuracy_wide_clean SWA

Validation accuracy 94.03% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/accuracy_wide_clean_swa.json`.

## 2026-09-18T10:20:30.592091+00:00 - Freeze final recipe

Selected `0.3250*accuracy_wide_clean_swa:10:T1.5 + 0.3250*accuracy_p2_clean_swa:30:T1.25 + 0.3500*mixer_pruned_75:10:T1.0` from 506 validation-only candidates: 5717/6000 correct (95.28%). Total stored parameters 2,207,556; dense MACs/image 3,183,978,880, including all members and views. Checkpoint hashes and the selection rule are in `results/final_recipe.json`. No test labels were loaded by selection.

## 2026-09-18T10:21:19.570613+00:00 - Frozen accuracy-first test evaluation

The frozen classifier correctly classified 9,439 of 10,000 test images (94.39%), with macro F1 0.943753 and negative log-likelihood 0.165926. Evidence: `results/final_test_metrics.json`. The recipe was frozen before this evaluation; no subsequent model, view, temperature or weight changes were made from test feedback.

Six additional training recipes, expanded inference views and complementary calibrated predictions produced the selected result. Larger models alone did not consistently improve validation accuracy, and the pruned checkpoint remained useful for complementary predictions despite having no compute advantage in the selection objective. These combined changes are not isolated causal ablations. The result improves on the initial 94.04% delivery by 35 correctly classified images, but does not establish statistical significance or guarantee the highest class ranking. The public benchmark had already been evaluated during development, so this remains exploratory evidence rather than an independently blinded test.

Delivery checks reloaded all 43 checkpoints and recomputed the frozen validation ensemble. The portable notebook executed successfully with six figures; the ZIP and saved prediction metrics passed verification. The included Section A PDF remains a typed study guide and must be replaced with the student's genuine handwritten scan before Canvas submission.

## 2026-09-18T10:30:19Z - Garment bottleneck and deployment follow-up

The user requested improved garment discrimination, an NVFP4 acceleration experiment, and selection aligned with the class bonus: accuracy first, then parameter count and/or FLOPs for ties. The existing classifier remains the reference. Four matched clean-refinement trials compare auxiliary garment conditional cross-entropy (weight 0.5) and gamma-one focal loss on the wide and fine-patch Mixers. Their clean-refinement controls already exist. Training uses only the fixed 54,000 training examples; selection uses the fixed 6,000 validation examples. The hypothesis follows previously inspected public test errors and is explicitly exploratory.

The bounded validation search considers ordinary/EMA/SWA checkpoints with 1/10/18/30 views, five calibration temperatures, and fixed mixture weights. It also tests reducing views or removing existing ensemble members. Selection maximizes validation accuracy, breaks ties by lower dense MACs then parameters then NLL, and guards both validation halves and Shirt F1. A failed targeted method is retained as a negative result. NVFP4 is a separate hardware deployment experiment: it changes numerical representation and possibly runtime/storage, not parameter count or mathematical dense FLOPs. Existing test artifacts will be preserved before any accepted replacement.

### NVFP4 execution and benchmark findings

`timeout 1800 .venv/bin/python benchmark_nvfp4.py --compile` exercised packed W4A4 channel Linear layers using torchao 0.16.0 and PyTorch 2.10.0 on RTX 5090. Profiling recorded `aten::_scaled_mm` and the SM120 CUTLASS E2M1 block-scaled GEMM kernel. Quantized weights occupy two FP4 values per byte; unsupported token dimensions, embeddings, norms and classifier heads retain the original precision. This is real hardware FP4 execution, not fake quantization or weight-only dequantization to BF16 GEMM.

The first compiled attempt reached PyTorch's default eight-specialization limit while comparing six models and multiple view layouts; partial evidence is in `results/nvfp4_initial_attempt.json`. The benchmark now permits 32 expected specializations. The completed run records synchronized batch-128 medians, all samples, storage bytes, kernel evidence and full validation for eager/compiled original/NVFP4 paths in `results/nvfp4_benchmark.json`. Compilation time and dataset/checkpoint loading are excluded; image transforms, inference, probability aggregation and CPU output transfer are included. No training experiment ran concurrently with timing; another idle service occupied GPU memory.

Original eager validation gives 5,717/6,000 correct, versus 5,694 for eager NVFP4. Compiled original gives 5,716 and compiled NVFP4 gives 5,693; compilation changes floating-point execution slightly and is not assumed bit-exact. NVFP4 is slower than the corresponding original-precision path here, so it is not promoted. Logical parameter counts and dense FLOPs do not decrease from quantization; byte storage and measured runtime are reported separately. No test-set quantization search was performed.

## 2026-09-18T10:31:38.090198+00:00 — garment_wide

Hypothesis: Matched to accuracy_wide_clean except an auxiliary conditional cross-entropy among T-shirt, Pullover, Dress, Coat and Shirt. Test whether training attention to garment distinctions improves validation without changing class priors.

Measured validation accuracy: 94.05%; checkpoint epoch 20; 1,297,746 parameters; 77,222,784 dense MACs/image; 74.0s training/validation wall time. Configuration and every epoch: `results/garment_wide.json`. Test set not evaluated in this run.

### 2026-09-18T10:31:38.090198+00:00 - garment_wide EMA

Validation accuracy 94.08% at epoch 20. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/garment_wide_ema.json`.

### 2026-09-18T10:31:38.090198+00:00 - garment_wide SWA

Validation accuracy 93.97% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/garment_wide_swa.json`.

## 2026-09-18T10:33:32.807238+00:00 — garment_p2

Hypothesis: Matched to accuracy_p2_clean except auxiliary conditional garment cross-entropy weight 0.5. Compare global accuracy and Shirt precision/recall on validation; retain control if no gain.

Measured validation accuracy: 94.05%; checkpoint epoch 10; 529,858 parameters; 72,329,664 dense MACs/image; 67.9s training/validation wall time. Configuration and every epoch: `results/garment_p2.json`. Test set not evaluated in this run.

### 2026-09-18T10:33:32.807238+00:00 - garment_p2 EMA

Validation accuracy 93.95% at epoch 11. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/garment_p2_ema.json`.

### 2026-09-18T10:33:32.807238+00:00 - garment_p2 SWA

Validation accuracy 93.80% at epoch 29. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/garment_p2_swa.json`.

## 2026-09-18T10:34:48.848052+00:00 — focal_wide

Hypothesis: Matched to accuracy_wide_clean except gamma-one focal loss. Downweight easy training examples without using validation or test labels as training targets; assess all classes to detect regressions.

Measured validation accuracy: 94.07%; checkpoint epoch 20; 1,297,746 parameters; 77,222,784 dense MACs/image; 73.5s training/validation wall time. Configuration and every epoch: `results/focal_wide.json`. Test set not evaluated in this run.

### 2026-09-18T10:34:48.848052+00:00 - focal_wide EMA

Validation accuracy 94.08% at epoch 20. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/focal_wide_ema.json`.

### 2026-09-18T10:34:48.848052+00:00 - focal_wide SWA

Validation accuracy 93.90% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/focal_wide_swa.json`.

## 2026-09-18T10:35:55.556835+00:00 — focal_p2

Hypothesis: Matched to accuracy_p2_clean except gamma-one focal loss. Test whether harder-example emphasis transfers to finer patches; select checkpoints using unchanged global validation accuracy and CE.

Measured validation accuracy: 93.82%; checkpoint epoch 10; 529,858 parameters; 72,329,664 dense MACs/image; 64.3s training/validation wall time. Configuration and every epoch: `results/focal_p2.json`. Test set not evaluated in this run.

### 2026-09-18T10:35:55.556835+00:00 - focal_p2 EMA

Validation accuracy 93.78% at epoch 12. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/focal_p2_ema.json`.

### 2026-09-18T10:35:55.556835+00:00 - focal_p2 SWA

Validation accuracy 93.70% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/focal_p2_swa.json`.

## 2026-09-18T10:36:42.903406+00:00 - Garment refinement and scoring-rule selection

Validation: 5717 -> 5717/6000 correct; Shirt F1 0.859589 -> 0.859589. Dense MACs/image 3,183,978,880 -> 3,183,978,880. Searched 583 recorded recipes; accuracy ranks first, with lower MACs/parameters breaking ties. Both validation halves and Shirt F1 were guarded. No new test evaluation was used for this selection. Evidence: `results/garment_recipe.json`, `results/garment_candidates.csv`.

## 2026-09-18T10:44:23.902436+00:00 — spatial_wide

Hypothesis: Replace mean pooling with a location-specific linear classifier, initialized to reproduce the old mean-pooled classifier. Matched to accuracy_wide_clean except this spatial readout; learn neckline/sleeve/body distinctions without CNN or attention.

Measured validation accuracy: 93.88%; checkpoint epoch 18; 1,389,906 parameters; 77,314,944 dense MACs/image; 71.6s training/validation wall time. Configuration and every epoch: `results/spatial_wide.json`. Test set not evaluated in this run.

### 2026-09-18T10:44:23.902436+00:00 - spatial_wide EMA

Validation accuracy 93.93% at epoch 6. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_wide_ema.json`.

### 2026-09-18T10:44:23.902436+00:00 - spatial_wide SWA

Validation accuracy 93.72% at epoch 27. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_wide_swa.json`.

## 2026-09-18T10:45:30.841363+00:00 — spatial_p2

Hypothesis: Matched to accuracy_p2_clean except a spatial linear readout initialized from repeated average-pooling classifier weights. Test whether retaining fine patch positions improves garment discrimination.

Measured validation accuracy: 93.72%; checkpoint epoch 16; 717,058 parameters; 72,516,864 dense MACs/image; 64.5s training/validation wall time. Configuration and every epoch: `results/spatial_p2.json`. Test set not evaluated in this run.

### 2026-09-18T10:45:30.841363+00:00 - spatial_p2 EMA

Validation accuracy 93.78% at epoch 8. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_p2_ema.json`.

### 2026-09-18T10:45:30.841363+00:00 - spatial_p2 SWA

Validation accuracy 93.72% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_p2_swa.json`.

## 2026-09-18T10:46:23.472102+00:00 — spatial_pruned

Hypothesis: Add a spatial readout to the compact pruned model and clean fine-tune. This is an architecture-plus-extra-training recipe trial; compare with spatial_pruned_garment for the isolated auxiliary-loss effect.

Measured validation accuracy: 93.67%; checkpoint epoch 2; 441,392 parameters; 24,247,552 dense MACs/image; 50.2s training/validation wall time. Configuration and every epoch: `results/spatial_pruned.json`. Test set not evaluated in this run.

### 2026-09-18T10:46:23.472102+00:00 - spatial_pruned EMA

Validation accuracy 93.62% at epoch 2. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_pruned_ema.json`.

### 2026-09-18T10:46:23.472102+00:00 - spatial_pruned SWA

Validation accuracy 93.37% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_pruned_swa.json`.

## 2026-09-18T10:47:17.290909+00:00 — spatial_pruned_garment

Hypothesis: Matched to spatial_pruned except auxiliary garment conditional cross-entropy. Retain all ten training classes during targeted post-training to avoid forgetting other categories.

Measured validation accuracy: 93.67%; checkpoint epoch 1; 441,392 parameters; 24,247,552 dense MACs/image; 51.2s training/validation wall time. Configuration and every epoch: `results/spatial_pruned_garment.json`. Test set not evaluated in this run.

### 2026-09-18T10:47:17.290909+00:00 - spatial_pruned_garment EMA

Validation accuracy 93.62% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_pruned_garment_ema.json`.

### 2026-09-18T10:47:17.290909+00:00 - spatial_pruned_garment SWA

Validation accuracy 93.43% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/spatial_pruned_garment_swa.json`.

## 2026-09-18T10:47:58.530022+00:00 - Spatial refinement and scoring-rule selection

Validation: 5717 -> 5717/6000 correct; Shirt F1 0.859589 -> 0.859589. Dense MACs/image 3,183,978,880 -> 3,183,978,880. Searched 583 recorded recipes; accuracy ranks first, with lower MACs/parameters breaking ties. Both validation halves and Shirt F1 were guarded. No new test evaluation was used for this selection. Evidence: `results/spatial_recipe.json`, `results/spatial_candidates.csv`.

## 2026-09-18T10:48:16.184783+00:00 - Confidence-gated inference

Tested 43 validation recipes. Selected 5721/6000 correct, Shirt F1 0.861063, 2945 early exits, 1,698,398,697 validation-average MACs and 3,261,201,664 worst-case MACs/image. Recomputed the actual routed pipeline before freezing; no test labels were loaded. The gate uses only model confidence and predicted non-garment class, and shares existing weights.

## 2026-09-18T10:49:36.089909+00:00 - Frozen cascade test outcome

After all 67 checkpoints and the actual routed validation pipeline passed verification, the frozen cascade correctly classified 9,442/10,000 test images (94.42%), with macro F1 0.944031. Relative to the preserved ungated reference, 16 predictions changed: eight errors were corrected and five correct predictions became errors, for a net gain of three. This is a small observed gain, not evidence of statistical significance. No model or threshold changes followed test evaluation.

The test set had 4,913 early exits. Average dense MACs/image were 1,696,912,840, 46.7% below the ungated reference; logical stored parameters remained 2,207,556. Worst-case MACs are 3,261,201,664 and must not be confused with the average. The separately timed 1,024-image validation batch took 235.9 ms for the cascade versus 388.7 ms for the full ensemble (1.65x throughput) on the shared RTX 5090. Evidence: `results/final_test_metrics.json`, `results/cascade_benchmark.json`, and preserved `results/pre_cascade_test_metrics.json` / predictions.

Shirt test recall remains 81.4% and F1 remains 0.828499. The eight targeted post-training/readout trials did not solve that class bottleneck; their negative results are retained. The deployed improvement is efficient conditional inference with slightly higher observed overall accuracy. All fitting stayed within Fashion-MNIST; no MNIST or external pretraining was introduced. The notebook includes the complete follow-up, actual inference and honest limitations; the typed Section A guide still requires replacement by a genuine handwritten scan.

## 2026-09-18T11:37:37.618043+00:00 — yolo_pyramid_control

Hypothesis: Pure MLP hierarchy control: 14x14 fine tokens, adjacent 2x2 concatenation and Linear projection to 7x7 coarse tokens, coarse pooled classification only. Same training budget as the feature-fusion variant; Fashion-MNIST only.

Measured validation accuracy: 93.02%; checkpoint epoch 112; 826,153 parameters; 68,717,952 dense MACs/image; 276.6s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_control.json`. Test set not evaluated in this run.

### 2026-09-18T11:37:37.618043+00:00 - yolo_pyramid_control EMA

Validation accuracy 92.98% at epoch 112. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_control_ema.json`.

### 2026-09-18T11:37:37.618043+00:00 - yolo_pyramid_control SWA

Validation accuracy 93.07% at epoch 113. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_control_swa.json`.

## 2026-09-18T11:42:35.384828+00:00 — yolo_pyramid_fusion

Hypothesis: Matched to the pyramid coarse-only control except concatenate pooled fine and coarse features before classification. This tests multiscale feature fusion inspired by detector feature pyramids without convolution, attention or detection losses. Backbone initialization and training RNG are matched.

Measured validation accuracy: 92.60%; checkpoint epoch 88; 827,305 parameters; 68,718,912 dense MACs/image; 284.5s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_fusion.json`. Test set not evaluated in this run.

### 2026-09-18T11:42:35.384828+00:00 - yolo_pyramid_fusion EMA

Validation accuracy 92.83% at epoch 90. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_fusion_ema.json`.

### 2026-09-18T11:42:35.384828+00:00 - yolo_pyramid_fusion SWA

Validation accuracy 92.70% at epoch 100. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_fusion_swa.json`.

## 2026-09-18T11:47:28.375794+00:00 — yolo_pyramid_aux

Hypothesis: Matched to pyramid fusion except a training-only ten-class auxiliary head on pooled fine-stage features, CE weight 0.3. Head initialization preserves the training RNG; deployment removes the head. Tests intermediate supervision, not the full YOLOv9 PGI method.

Measured validation accuracy: 92.45%; checkpoint epoch 78; 827,305 parameters; 68,718,912 dense MACs/image; 290.6s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_aux.json`. Test set not evaluated in this run.

### 2026-09-18T11:47:28.375794+00:00 - yolo_pyramid_aux EMA

Validation accuracy 92.70% at epoch 57. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_aux_ema.json`.

### 2026-09-18T11:47:28.375794+00:00 - yolo_pyramid_aux SWA

Validation accuracy 92.42% at epoch 100. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_aux_swa.json`.

## 2026-09-18T11:52:37.077479+00:00 — yolo_pyramid_csp

Hypothesis: Replace full channel MLPs in the feature-fused pyramid with half-channel transforms, preserved half channels, concatenation and dense fusion. CSP-inspired pure MLP experiment; same seed and training hyperparameters but fewer actual parameters/MACs. Changed architecture consumes a different initialization RNG stream; not a multi-seed causal claim.

Measured validation accuracy: 92.80%; checkpoint epoch 79; 550,393 parameters; 47,042,880 dense MACs/image; 306.3s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_csp.json`. Test set not evaluated in this run.

### 2026-09-18T11:52:37.077479+00:00 - yolo_pyramid_csp EMA

Validation accuracy 92.92% at epoch 68. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_csp_ema.json`.

### 2026-09-18T11:52:37.077479+00:00 - yolo_pyramid_csp SWA

Validation accuracy 92.67% at epoch 98. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_csp_swa.json`.

## 2026-09-18T11:53:45.854378+00:00 — yolo_pyramid_control_clean

Hypothesis: Matched 30-epoch clean refinement of yolo_pyramid_control; initialize its best raw-validation ordinary/EMA/SWA checkpoint. Same clean schedule and learning rate for all four architectures; auxiliary supervision remains training-only when configured.

Measured validation accuracy: 93.50%; checkpoint epoch 1; 826,153 parameters; 68,717,952 dense MACs/image; 66.3s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_control_clean.json`. Test set not evaluated in this run.

### 2026-09-18T11:53:45.854378+00:00 - yolo_pyramid_control_clean EMA

Validation accuracy 93.43% at epoch 4. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_control_clean_ema.json`.

### 2026-09-18T11:53:45.854378+00:00 - yolo_pyramid_control_clean SWA

Validation accuracy 93.15% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_control_clean_swa.json`.

## 2026-09-18T11:54:56.637924+00:00 — yolo_pyramid_fusion_clean

Hypothesis: Matched 30-epoch clean refinement of yolo_pyramid_fusion; initialize its best raw-validation ordinary/EMA/SWA checkpoint. Same clean schedule and learning rate for all four architectures; auxiliary supervision remains training-only when configured.

Measured validation accuracy: 93.30%; checkpoint epoch 2; 827,305 parameters; 68,718,912 dense MACs/image; 68.4s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_fusion_clean.json`. Test set not evaluated in this run.

### 2026-09-18T11:54:56.637924+00:00 - yolo_pyramid_fusion_clean EMA

Validation accuracy 93.27% at epoch 6. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_fusion_clean_ema.json`.

### 2026-09-18T11:54:56.637924+00:00 - yolo_pyramid_fusion_clean SWA

Validation accuracy 93.12% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_fusion_clean_swa.json`.

## 2026-09-18T11:56:08.844221+00:00 — yolo_pyramid_aux_clean

Hypothesis: Matched 30-epoch clean refinement of yolo_pyramid_aux; initialize its best raw-validation ordinary/EMA/SWA checkpoint. Same clean schedule and learning rate for all four architectures; auxiliary supervision remains training-only when configured.

Measured validation accuracy: 93.20%; checkpoint epoch 1; 827,305 parameters; 68,718,912 dense MACs/image; 69.8s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_aux_clean.json`. Test set not evaluated in this run.

### 2026-09-18T11:56:08.844221+00:00 - yolo_pyramid_aux_clean EMA

Validation accuracy 93.15% at epoch 13. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_aux_clean_ema.json`.

### 2026-09-18T11:56:08.844221+00:00 - yolo_pyramid_aux_clean SWA

Validation accuracy 93.03% at epoch 28. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_aux_clean_swa.json`.

## 2026-09-18T11:57:25.149024+00:00 — yolo_pyramid_csp_clean

Hypothesis: Matched 30-epoch clean refinement of yolo_pyramid_csp; initialize its best raw-validation ordinary/EMA/SWA checkpoint. Same clean schedule and learning rate for all four architectures; auxiliary supervision remains training-only when configured.

Measured validation accuracy: 93.30%; checkpoint epoch 3; 550,393 parameters; 47,042,880 dense MACs/image; 73.9s training/validation wall time. Configuration and every epoch: `results/yolo_pyramid_csp_clean.json`. Test set not evaluated in this run.

### 2026-09-18T11:57:25.149024+00:00 - yolo_pyramid_csp_clean EMA

Validation accuracy 93.33% at epoch 4. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_csp_clean_ema.json`.

### 2026-09-18T11:57:25.149024+00:00 - yolo_pyramid_csp_clean SWA

Validation accuracy 92.67% at epoch 25. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/yolo_pyramid_csp_clean_swa.json`.

## 2026-09-18T15:40:17.558232+00:00 — features_gray

Hypothesis: Matched 30-epoch clean refinement from the same wide SWA parent. gray: retain grayscale; edges add signed central differences, contrast additionally adds grayscale minus replicate-padded 3x3 mean. Identical seed and training budget; new input weights start at zero. Select on validation, without accessing test.

Measured validation accuracy: 94.03%; checkpoint epoch 2; 1,297,746 parameters; 77,222,784 dense MACs/image; 72.2s training/validation wall time. Configuration and every epoch: `results/features_gray.json`. Test set not evaluated in this run.

### 2026-09-18T15:40:17.558232+00:00 - features_gray EMA

Validation accuracy 94.02% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_gray_ema.json`.

### 2026-09-18T15:40:17.558232+00:00 - features_gray SWA

Validation accuracy 93.90% at epoch 27. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_gray_swa.json`.

## 2026-09-18T15:41:32.608443+00:00 — features_edges

Hypothesis: Matched 30-epoch clean refinement from the same wide SWA parent. edges: retain grayscale; edges add signed central differences, contrast additionally adds grayscale minus replicate-padded 3x3 mean. Identical seed and training budget; new input weights start at zero. Select on validation, without accessing test.

Measured validation accuracy: 94.03%; checkpoint epoch 2; 1,303,890 parameters; 77,523,840 dense MACs/image; 72.7s training/validation wall time. Configuration and every epoch: `results/features_edges.json`. Test set not evaluated in this run.

### 2026-09-18T15:41:32.608443+00:00 - features_edges EMA

Validation accuracy 94.03% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_edges_ema.json`.

### 2026-09-18T15:41:32.608443+00:00 - features_edges SWA

Validation accuracy 93.92% at epoch 30. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_edges_swa.json`.

## 2026-09-18T15:42:47.476393+00:00 — features_contrast

Hypothesis: Matched 30-epoch clean refinement from the same wide SWA parent. contrast: retain grayscale; edges add signed central differences, contrast additionally adds grayscale minus replicate-padded 3x3 mean. Identical seed and training budget; new input weights start at zero. Select on validation, without accessing test.

Measured validation accuracy: 94.00%; checkpoint epoch 2; 1,306,962 parameters; 77,674,368 dense MACs/image; 72.5s training/validation wall time. Configuration and every epoch: `results/features_contrast.json`. Test set not evaluated in this run.

### 2026-09-18T15:42:47.476393+00:00 - features_contrast EMA

Validation accuracy 94.00% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_contrast_ema.json`.

### 2026-09-18T15:42:47.476393+00:00 - features_contrast SWA

Validation accuracy 93.95% at epoch 29. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/features_contrast_swa.json`.

## 2026-09-18T15:55:33.147074+00:00 — moe_control

Hypothesis: Capacity control: one width-1152 residual MLP head versus three width-384 heads. Same parent, 30 clean epochs, seed and trunk dropout/shuffle stream; expert branches use no dropout. Approximately equal total parameters, not equal active MACs. Zero residual output preserves parent initialization.

Measured validation accuracy: 94.00%; checkpoint epoch 2; 1,741,651 parameters; 77,665,344 dense MACs/image; 77.7s training/validation wall time. Configuration and every epoch: `results/moe_control.json`. Test set not evaluated in this run.

### 2026-09-18T15:55:33.147074+00:00 - moe_control EMA

Validation accuracy 94.03% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/moe_control_ema.json`.

### 2026-09-18T15:55:33.147074+00:00 - moe_control SWA

Validation accuracy 93.90% at epoch 28. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/moe_control_swa.json`.

## 2026-09-18T15:56:57.475704+00:00 — moe_mixer

Hypothesis: Matched 30-epoch clean fine-tuning from the frozen wide SWA parent. A shared MLP-Mixer trunk feeds three residual MLP experts and a learned top-1 router at inference; training uses soft routing. Expert residual outputs start at zero so the initialization matches the parent. Validation selects the ordinary, EMA or SWA checkpoint; no test labels are used.

Measured validation accuracy: 93.98%; checkpoint epoch 1; 1,742,421 parameters; 77,370,816 dense MACs/image; 81.9s training/validation wall time. Configuration and every epoch: `results/moe_mixer.json`. Test set not evaluated in this run.

### 2026-09-18T15:56:57.475704+00:00 - moe_mixer EMA

Validation accuracy 94.02% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/moe_mixer_ema.json`.

### 2026-09-18T15:56:57.475704+00:00 - moe_mixer SWA

Validation accuracy 93.73% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/moe_mixer_swa.json`.

## 2026-09-18T15:58:57.989886+00:00 — loss_focal2

Hypothesis: Paired against features_gray CE: only change loss to focal gamma=2. Same wide SWA parent and 30-epoch clean schedule; no class weighting or augmentation.

Measured validation accuracy: 94.00%; checkpoint epoch 1; 1,297,746 parameters; 77,222,784 dense MACs/image; 72.9s training/validation wall time. Configuration and every epoch: `results/loss_focal2.json`. Test set not evaluated in this run.

### 2026-09-18T15:58:57.989886+00:00 - loss_focal2 EMA

Validation accuracy 94.07% at epoch 1. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/loss_focal2_ema.json`.

### 2026-09-18T15:58:57.989886+00:00 - loss_focal2 SWA

Validation accuracy 93.85% at epoch 28. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/loss_focal2_swa.json`.

## 2026-09-18T16:00:13.911245+00:00 — loss_weighted

Hypothesis: Paired against features_gray CE: only weight T-shirt, Pullover, Coat and Shirt by 1.5, all others 1. Normalize by sum of sample weights, matching native weighted CE. Same parent, seed and 30 clean epochs. This is difficulty weighting; Fashion-MNIST classes are balanced.

Measured validation accuracy: 94.02%; checkpoint epoch 2; 1,297,746 parameters; 77,222,784 dense MACs/image; 73.4s training/validation wall time. Configuration and every epoch: `results/loss_weighted.json`. Test set not evaluated in this run.

### 2026-09-18T16:00:13.911245+00:00 - loss_weighted EMA

Validation accuracy 94.02% at epoch 2. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/loss_weighted_ema.json`.

### 2026-09-18T16:00:13.911245+00:00 - loss_weighted SWA

Validation accuracy 93.97% at epoch 26. Same training trajectory as the ordinary checkpoint; training cost is shared, not an independent run. EMA decay 0.995 after each batch; SWA snapshots at each epoch in the final 20%. LayerNorm needs no BatchNorm recalibration. Evidence: `results/loss_weighted_swa.json`.
