"""Model registry shared by the evaluation scripts (retrained models live in ../checkpoints)."""
import os, torch
from common import *

# name -> architecture flags of the retrained models
CFG = {
    "r18_plain":     dict(depth=18, blur=0, denoise=0, avgmax=0, act="relu"),
    "r18_full":      dict(depth=18, blur=1, denoise=1, avgmax=1, act="relu"),
    "r18_noblur":    dict(depth=18, blur=0, denoise=1, avgmax=1, act="relu"),
    "r18_nodenoise": dict(depth=18, blur=1, denoise=0, avgmax=1, act="relu"),
    "r18_noavgmax":  dict(depth=18, blur=1, denoise=1, avgmax=0, act="relu"),
    "r18v2_plain":   dict(depth=18, blur=0, denoise=0, avgmax=0, act="silu"),
    "r18v2_full":    dict(depth=18, blur=1, denoise=1, avgmax=1, act="silu"),
    "r18v2_trades":  dict(depth=18, blur=1, denoise=1, avgmax=1, act="silu"),
    "r34_plain":     dict(depth=34, blur=0, denoise=0, avgmax=0, act="relu"),
    "r34_full":      dict(depth=34, blur=1, denoise=1, avgmax=1, act="relu"),
}


def build_new(name, which="best"):
    c = CFG[name]
    net = SecureNet(c["depth"], bool(c["blur"]), bool(c["denoise"]), bool(c["avgmax"]), c["act"])
    net.load_state_dict(torch.load(os.path.join(HERE_CKPT, f"{name}_{which}.pt"), map_location=DEVICE))
    return Normalized(net, CIFAR_MEAN, CIFAR_STD).to(DEVICE).eval()


HERE_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

# the three originally trained models (read from the project folder)
OLD = {
    "Baseline CNN":    lambda: wrap01(load_baseline()),
    "SecureCNN (orig.)":     lambda: wrap01(load_secure("best_securecnn.pth", 18)),
    "SecureResNet34 (orig.)": lambda: wrap01(load_secure("secure_resnet34_pretrained_pgd3_epoch_15.pth", 34)),
}
