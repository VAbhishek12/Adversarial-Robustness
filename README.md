# Do Smoothing and Denoising Modules Improve Adversarial Training? A Controlled Study on CIFAR-10

Code and results behind the M.Tech thesis and the accompanying research paper (Vedula Abhishek, C-DAC Mohali, I.K. Gujral Punjab Technical University; supervisor Ms. Sonia Dosanjh).

The study asks whether three architectural parts that are often added to adversarially trained CNNs (a frozen Gaussian smoothing layer, a residual denoising block and an average plus maximum pooling head) improve robustness on CIFAR-10 when every network is trained with one shared recipe and tested with strong attacks in pixel space. It also documents how a perturbation budget applied to a normalised input makes a defence look about four times stronger than it is.

## Main findings

All values are accuracy in percent on the CIFAR-10 test set. PGD-20 uses `eps = 8/255` in pixel space on all 10 000 test images. APGD is the worst case of APGD-CE and APGD-DLR on the first 1 000 test images.

**Networks trained with one shared recipe (ResNet-18, 20 epochs)**

| Network | Clean | PGD-20 | APGD | Params (M) |
|---|---|---|---|---|
| ResNet-18 plain | 78.06 | 49.18 | 44.7 | 11.69 |
| ResNet-18 with all three parts | 75.69 | 46.61 | 42.0 | 16.41 |
| ResNet-18 without denoising block | 75.82 | 46.55 | n/a | 11.69 |
| ResNet-18 without smoothing layer | 76.14 | 47.32 | n/a | 16.41 |
| ResNet-18 plain, SiLU and weight averaging | 73.57 | 47.74 | 44.1 | 11.69 |

**The three original models under the two evaluation conventions**

| Network | Clean | Normalised budget 0.03 (PGD) | Pixel budget 8/255 (PGD-20) | Pixel budget 8/255 (APGD) |
|---|---|---|---|---|
| Baseline CNN | 74.79 | 9.40 | 0.29 | 0.00 |
| SecureCNN | 90.24 | 59.07 | 5.96 | 0.60 |
| SecureResNet34 | 83.00 | 61.55 | 0.82 | 0.20 |

The plain adversarially trained ResNet-18 is the strongest design in these experiments. Removing the denoising block or the smoothing layer on its own does not restore its accuracy. Further metrics (precision, recall, F1, MCC, Cohen's kappa, AUROC, NLL, ECE, attack success rate, critical budget, gradient norm) are in `results/metrics.json` and `results/SUMMARY.md`.

## Repository layout

```
.
├── experiments/            code that produced the reported numbers
│   ├── common.py           models (BaselineCNN, SecureNet), data loaders, FGSM and PGD in both conventions
│   ├── registry.py         names and architecture flags of the retrained networks
│   ├── metrics_lib.py      accuracy, precision, recall, F1, MCC, kappa, AUROC, NLL, ECE, ASR, ...
│   ├── train_at.py         PGD adversarial training in pixel space (the shared recipe)
│   ├── eval_new.py         staged evaluation of retrained networks (clean, PGD-20, budget sweep, APGD, cost)
│   ├── eval_metrics.py     full metric suite for every network
│   ├── exp0_identify.py    clean accuracy of every original checkpoint
│   ├── exp1_protocolA.py   original evaluation (budget on the normalised input), reproduces the first results
│   ├── exp2_protocolB.py   corrected evaluation in pixel space (FGSM, PGD-20, APGD-CE, APGD-DLR)
│   ├── exp3_compute.py     parameters, operations, size and latency
│   ├── exp7_equal_attack.py  equal-attack comparison of SecureCNN and SecureResNet34
│   └── summarize.py, ablation_summary.py   collect results into results/SUMMARY.md and print deltas
├── notebooks/original/     the original notebooks that trained and first evaluated the three original models
├── results/                every measurement behind the thesis and paper (JSON, CSV, logs)
├── checkpoints/            put model weights here (see checkpoints/README.md)
├── scripts/reproduce.sh    the exact command sequence
├── requirements.txt
└── CITATION.cff
```

## The three original models and the retrained networks

* **Baseline CNN**: three convolution blocks and two linear layers, trained with cross entropy (`notebooks/original/Baseline_CNN.ipynb`).
* **SecureCNN**: ResNet-18 with a frozen 3x3 Gaussian smoothing layer after the first convolution, a residual denoising block after the last stage and an average plus maximum pooling head, trained with PGD adversarial training on mixed clean and attacked batches (`SecureCNN_Adversarial_Defense.ipynb`).
* **SecureResNet34**: the same design on a ResNet-34 backbone (`SecureResNet34_Training_epoch15_log.ipynb`, `SecureResNet34_Training.ipynb`).
* **Retrained networks** (`experiments/train_at.py`): ResNet-18 backbones trained again with one recipe: PGD adversarial training in pixel space at `eps = 8/255`, three steps of size `2.5 * eps / 3` with a random start, attacked images only, SGD with Nesterov momentum 0.9, weight decay `5e-4`, batch size 128, one cycle learning rate with peak 0.1, 20 epochs, random crops and flips, bfloat16. One thousand training images are held out for validation, and the checkpoint with the best validation PGD-10 accuracy is tested. The test set plays no role in choosing a model. Switches: `--blur`, `--denoise`, `--avgmax`, `--act {relu,silu}`, `--ema`, `--seed` (default 0).

Two evaluation conventions are used in the code:

* **Protocol A** (`exp1_protocolA.py`): the original method, with `eps` applied to the normalised tensor and clamping to `[-1, 1]`.
* **Protocol B** (`exp2_protocolB.py`, `eval_new.py`): the corrected one, with `[0, 1]` pixel inputs, normalisation inside the model, and `eps` in units of 1/255.

## Setup

Tested with Python 3.13, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21.0 and torchattacks 3.5.1 on a laptop NVIDIA RTX 3050 (4 GB) under Windows 11. Any recent PyTorch with a CUDA GPU should work. Training and the attack evaluations are slow on a CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

CIFAR-10 is downloaded automatically by torchvision on first use, from the official source at [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) ("CIFAR-10 python version"). It is saved under `data/` by default; set the environment variable `CIFAR_DIR` to point at an existing copy instead.

On Windows, if PyTorch reports a duplicate OpenMP runtime, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Model weights

Weights are not stored in git because of their size (about 1.7 GB in total). `checkpoints/README.md` lists the file names the code expects. The original three checkpoints go in `checkpoints/original/` (or in the folder named by `ORIG_CKPT_DIR`). The retrained networks are written to `checkpoints/` by `train_at.py`.

## Reproducing the results

Run everything from the `experiments/` folder. `scripts/reproduce.sh` contains the full sequence. In short:

```bash
cd experiments

# 1. original models: reproduction, corrected evaluation, cost, equal-attack check
python exp0_identify.py
python exp1_protocolA.py
python exp2_protocolB.py
python exp3_compute.py
python exp7_equal_attack.py

# 2. retrained networks (about 30 to 80 minutes each on an RTX 3050)
python train_at.py --name r18_plain     --depth 18 --blur 0 --denoise 0 --avgmax 0 --epochs 20
python train_at.py --name r18_full      --depth 18 --blur 1 --denoise 1 --avgmax 1 --epochs 20
python train_at.py --name r18_nodenoise --depth 18 --blur 1 --denoise 0 --avgmax 1 --epochs 20
python train_at.py --name r18_noblur    --depth 18 --blur 0 --denoise 1 --avgmax 1 --epochs 20
python train_at.py --name r18v2_plain   --depth 18 --blur 0 --denoise 0 --avgmax 0 --act silu --ema 0.995 --epochs 20

# 3. evaluation (AA=1 adds the APGD worst case on 1 000 images, AA=0 skips it)
AA=1 python eval_new.py r18_plain r18_full r18v2_plain
AA=0 python eval_new.py r18_nodenoise r18_noblur
python eval_metrics.py

# 4. summary
python summarize.py
```

Notes on reproducibility:

* The training seed is fixed (`--seed 0`) and the validation split uses a fixed generator. GPU kernels, `cudnn.benchmark` and bfloat16 make runs non-deterministic at the level of small differences. Expect results within a few tenths of a point, not identical digits.
* Attacks use a random start. Numbers on 1 000 to 2 000 images carry a sampling error of about 1 to 1.6 points.
* `results/` holds the exact outputs used in the thesis and paper. `new_eval.json` and `metrics.json` are updated incrementally by the scripts, so a partial rerun does not delete finished entries.
* The original notebooks were run on a Windows machine and contain local paths. They are kept as the record of how the three original models were trained and first evaluated.

## Results files

| File | Content |
|---|---|
| `results/protocolA.json`, `protocolB.json` | accuracy of the three original models under both conventions |
| `results/equal_attack.json` | equal-attack comparison of the two secure models |
| `results/compute.json` | parameters, operations, size, latency |
| `results/new_eval.json`, `new_eval_partial.json` | retrained networks: clean, PGD-20, budget sweep, APGD, cost |
| `results/metrics.json` | full metric suite for all networks |
| `results/train_*.json`, `train_*.log` | per-epoch training records of the retrained networks |
| `results/training_logs.json` | training logs of the three original models (extracted from the notebook outputs) |
| `results/checkpoint_clean_acc.csv` | clean accuracy of every original checkpoint |
| `results/SUMMARY.md` | one summary table |

## Citation

If you use this code, please cite the thesis or paper (see `CITATION.cff`).
