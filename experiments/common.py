"""Shared models, data and attack helpers for the thesis re-evaluation.

Model definitions are copied verbatim from the project notebooks
(Baseline_CNN, SecureCNN_Adversarial_Defense, SecureResNet34_*). The
checkpoints are only ever read.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# folder with the three original checkpoints (see checkpoints/README.md) and the CIFAR-10 folder; both can be overridden
PROJECT = os.environ.get("ORIG_CKPT_DIR", os.path.join(REPO, "checkpoints", "original"))
DATA = os.environ.get("CIFAR_DIR", os.path.join(REPO, "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
HALF_MEAN, HALF_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ----------------------------------------------------------------- models
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class GaussianBlur(nn.Module):
    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
        self.weight = nn.Parameter(k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1),
                                   requires_grad=False)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.groups)


class DenoiseBlock(nn.Module):
    def __init__(self, c, act="relu"):
        super().__init__()
        self.act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class AvgMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg, self.max = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return self.avg(x) + self.max(x)


class SecureNet(nn.Module):
    """SecureCNN (depth=18) or SecureResNet34 (depth=34); flags allow ablations."""

    def __init__(self, depth=18, blur=True, denoise=True, avgmax=True, act="relu"):
        super().__init__()
        self.backbone = (resnet18 if depth == 18 else resnet34)(weights=None)
        if act == "silu":                      # SiLU/Swish everywhere (Gowal et al. 2020)
            self.backbone.relu = nn.SiLU()
            for layer in (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4):
                for blk in layer:
                    blk.relu = nn.SiLU()
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.gaussian = GaussianBlur(64) if blur else nn.Identity()
        self.denoise = DenoiseBlock(512, act) if denoise else nn.Identity()
        self.pool = AvgMaxPool() if avgmax else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = self.gaussian(x)
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        x = self.pool(self.denoise(x))
        return self.fc(torch.flatten(x, 1))


class Normalized(nn.Module):
    """Accepts [0,1] pixels; normalises inside so attacks live in pixel space."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def _load(model, ckpt):
    sd = torch.load(os.path.join(PROJECT, ckpt), map_location=DEVICE)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def load_baseline(ckpt="baseline_cnn_cifar10.pth"):
    return _load(BaselineCNN(), ckpt), HALF_MEAN, HALF_STD


def load_secure(ckpt, depth=18):
    return _load(SecureNet(depth), ckpt), CIFAR_MEAN, CIFAR_STD


def wrap01(model_mean_std):
    """Pixel-space ([0,1]) model for the corrected protocol."""
    m, mean, std = model_mean_std
    return Normalized(m, mean, std).to(DEVICE).eval()


# ------------------------------------------------------------------- data
def test_set(mean, std):
    ds = torchvision.datasets.CIFAR10(DATA, train=False, download=True,
                                      transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    return ds


def pixel_test_set():
    return torchvision.datasets.CIFAR10(DATA, train=False, download=True,
                                        transform=T.ToTensor())


def pixel_loader(batch=128, n=None):
    ds = pixel_test_set()
    if n:
        ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


# ------------------------------- Protocol A: exactly as in the notebooks
def fgsm_A(model, x, y, eps, lo=-1., hi=1.):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, x)[0]
    return torch.clamp(x + eps * g.sign(), lo, hi).detach()


def pgd_A(model, x, y, eps, alpha, steps, lo=-1., hi=1., rand_start=True):
    adv = x.clone().detach()
    if rand_start:
        adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), lo, hi).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(adv), y), adv)[0]
        adv = adv.detach() + alpha * g.sign()
        adv = torch.clamp(x + torch.clamp(adv - x, -eps, eps), lo, hi).detach()
    return adv


# --------------------------------- Protocol B: pixel space, correct [0,1]
def fgsm_B(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    g = torch.autograd.grad(F.cross_entropy(model(x), y), x)[0]
    return torch.clamp(x + eps * g.sign(), 0, 1).detach()


def pgd_B(model, x, y, eps, steps=20, alpha=None, restarts=1):
    alpha = alpha or 2.5 * eps / steps
    worst = x.clone()
    still_ok = torch.ones(len(x), dtype=torch.bool, device=x.device)
    for _ in range(restarts):
        adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            g = torch.autograd.grad(F.cross_entropy(model(adv), y), adv)[0]
            adv = adv.detach() + alpha * g.sign()
            adv = torch.clamp(x + torch.clamp(adv - x, -eps, eps), 0, 1).detach()
        with torch.no_grad():
            fooled = model(adv).argmax(1) != y
        take = fooled & still_ok
        worst[take] = adv[take]
        still_ok &= ~fooled
    worst[still_ok] = adv[still_ok]
    return worst


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().sum().item()
