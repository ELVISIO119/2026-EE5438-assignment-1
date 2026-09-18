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
