# Model weights

Weights are not part of the git repository (about 1.7 GB). Download or copy them into this folder before running the evaluation scripts. A good place to host them is a GitHub release, Zenodo or the Hugging Face Hub.

## Original models: `checkpoints/original/`

These three files are read by `experiments/common.py` and `experiments/registry.py`. Set `ORIG_CKPT_DIR` to use another folder.

| File | Model | Clean accuracy |
|---|---|---|
| `baseline_cnn_cifar10.pth` | Baseline CNN | 74.79 |
| `best_securecnn.pth` | SecureCNN (ResNet-18 with the three parts, adversarially trained, epoch 100) | 90.24 |
| `secure_resnet34_pretrained_pgd3_epoch_15.pth` | SecureResNet34 (epoch 15 checkpoint) | 83.01 |

`experiments/exp0_identify.py` reports the clean accuracy of every `*.pth` file in the folder and writes `results/checkpoint_clean_acc.csv`.

## Retrained ResNet-18 networks: `checkpoints/`

Written by `experiments/train_at.py`. For each run it saves `<name>_best.pt` (best validation PGD-10, used for testing), `<name>_final.pt` (last epoch) and `<name>_state.pt` (resume state).

| Run name | Parts | Activation |
|---|---|---|
| `r18_plain` | none | ReLU |
| `r18_full` | smoothing layer, denoising block, average plus max pooling | ReLU |
| `r18_nodenoise` | smoothing layer, average plus max pooling | ReLU |
| `r18_noblur` | denoising block, average plus max pooling | ReLU |
| `r18v2_plain` | none, with weight averaging (decay 0.995) | SiLU |

Load one with:

```python
from registry import build_new
model = build_new("r18_plain", which="best")   # takes [0, 1] pixel inputs and normalises inside
```
