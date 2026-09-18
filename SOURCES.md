# Sources and implementation notes

- Assignment: https://eelmpo.github.io/ee5438/pdf/2026_EE5438_Ass01.pdf
- Course slides: https://eelmpo.github.io/ee5438/schedule.html
- Starter notebook: https://colab.research.google.com/drive/1UQYf7m0xGBaqUgnviLBR9rfbmddkUsxq
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Muon parameter-group guidance: https://github.com/KellerJordan/Muon (README checked 2026-09-18).
- Native Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- Native parameter averaging: https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema
- MLP-Mixer architecture: https://arxiv.org/abs/2105.01601
- SAM: https://arxiv.org/abs/2010.01412
- ASAM: https://arxiv.org/abs/2102.11600
- Knowledge distillation: https://arxiv.org/abs/1503.02531
- Temperature scaling: Guo et al., *On Calibration of Modern Neural Networks* (2017), https://arxiv.org/abs/1706.04599 (official title/authors checked 2026-09-18). Here temperature is applied to log probabilities after averaging inference views; this is an assignment-specific use, not a reproduction of the paper's experiments.
- Focal loss: Lin et al., *Focal Loss for Dense Object Detection* (2017), https://arxiv.org/abs/1708.02002 (official title/authors checked 2026-09-18). The follow-up uses gamma 1, no class-balancing alpha, and multiclass MLP classification; it does not reproduce the paper's object-detection setting. Auxiliary garment conditional cross-entropy is an assignment-specific experiment.
- Real NVFP4 quantization and inference: pinned torchao 0.16.0 [configuration/shape constraints](https://github.com/pytorch/ao/blob/v0.16.0/torchao/prototype/mx_formats/inference_workflow.py), [packed tensors and native scaled GEMM](https://github.com/pytorch/ao/blob/v0.16.0/torchao/prototype/mx_formats/nvfp4_tensor.py), and [PyTorch version compatibility](https://github.com/pytorch/ao/blob/v0.16.0/torchao/__init__.py), checked 2026-09-18. The experiment uses two FP4 values per byte, blocks of 16, FP8 block scales and FP32 tensor scales. Weight-only dequantization to a higher-precision GEMM is not presented as W4A4 hardware acceleration.

The models are small assignment-specific implementations trained from scratch on the official training split. No pretrained weights or external training examples are used. Native PyTorch implements AdamW, Muon, averaging, loss functions and automatic differentiation; the small SAM/ASAM perturbation helper has a runnable restoration check. This is a single-seed exploratory benchmark, not a statistical replication of the cited papers.

Computational complexity is reported as dense multiply-accumulate operations (MACs) per image. Approximate dense FLOPs = 2 x MACs. Normalization, activations, softmax, augmentation and memory traffic are excluded; parameter counts include biases and normalization gains. Timing is observed wall time on a shared RTX 5090, not a dedicated-device speed claim.
