"""Parameters, size, FLOPs, latency -> results/compute.json (fixes the inconsistent
params column in the original paper: total == trainable + frozen)."""
import json, time, os, torch
from torch.utils.flop_counter import FlopCounterMode
from common import *

MODELS = {
    "Baseline CNN":   (lambda: load_baseline(), "baseline_cnn_cifar10.pth"),
    "SecureCNN":      (lambda: load_secure("best_securecnn.pth", 18), "best_securecnn.pth"),
    "SecureResNet34": (lambda: load_secure("secure_resnet34_pretrained_pgd3_epoch_15.pth", 34),
                       "secure_resnet34_pretrained_pgd3_epoch_15.pth"),
}
out = {}
for name, (fn, ck) in MODELS.items():
    m = wrap01(fn())
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x1 = torch.rand(1, 3, 32, 32, device=DEVICE)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x1)
    macs = fc.get_total_flops() / 2
    with torch.no_grad():
        for _ in range(30): m(x1)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(300): m(x1)
        torch.cuda.synchronize(); lat = (time.perf_counter() - t) / 300 * 1000
    inner = m.model
    if name == "Baseline CNN":
        parts = {"Conv1 (3-32)": inner.features[0], "Conv2 (32-64)": inner.features[3],
                 "Conv3 (64-128)": inner.features[6], "FC 2048-256": inner.classifier[0],
                 "FC 256-10": inner.classifier[2]}
    else:
        b = inner.backbone
        parts = {"Stem conv + BN": torch.nn.ModuleList([b.conv1, b.bn1]),
                 "Gaussian blur (frozen)": inner.gaussian,
                 "Stage 1 (64 ch)": b.layer1, "Stage 2 (128 ch)": b.layer2,
                 "Stage 3 (256 ch)": b.layer3, "Stage 4 (512 ch)": b.layer4,
                 "Denoise block (512 ch)": inner.denoise, "Classifier FC 512-10": inner.fc}
    breakdown = {k: dict(total=sum(p.numel() for p in v.parameters()),
                         trainable=sum(p.numel() for p in v.parameters() if p.requires_grad))
                 for k, v in parts.items()}
    out[name] = dict(breakdown=breakdown,
                     params_total_M=total / 1e6, params_trainable_M=train / 1e6,
                     ckpt_MB=os.path.getsize(os.path.join(PROJECT, ck)) / 2**20,
                     MACs_M=macs / 1e6, latency_ms_bs1=lat)
    print(name, out[name], flush=True)
json.dump(out, open("../results/compute.json", "w"), indent=1)
