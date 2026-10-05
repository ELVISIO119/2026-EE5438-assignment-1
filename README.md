# EE5438 Assignment 1

Student: Cai Haochen (`58561440`)

This branch contains one self-contained notebook for the pure-MLP
Fashion-MNIST assignment. The notebook visibly defines the model classes, data
split, baseline SGD training, AdamW/Muon training loop, augmentation,
checkpoint selection, structured pruning, clean fine-tuning and validation-only
route search.

Run with:

```bash
python -m pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute Assign01_Cai_Haochen_58561440.ipynb --output executed.ipynb
```

`RETRAIN_ALL=False` verifies the frozen checkpoints and final result. Set it
to `True` to replay the complete training sequence from the notebook. The
recorded endpoint is 94.55% test accuracy, 3,164,836 parameters and
301,330,871 average MACs/image. No CNN, Transformer, attention, pretrained
weights or external images are used.
