"""Protocol A: reproduce the paper/notebook numbers.
eps is applied in *normalised* space and adversarial images are clamped to [-1,1],
exactly as in the original notebooks. Output: results/protocolA.json
"""
import json, time, torch
from common import *

torch.manual_seed(0)
MODELS = {
    "Baseline CNN":   (lambda: load_baseline(), dict(steps=7, alpha=lambda e: e / 7, rand=False)),  # baseline notebook: no random start
    "SecureCNN":      (lambda: load_secure("best_securecnn.pth", 18), dict(steps=10, alpha=lambda e: e / 4)),
    "SecureResNet34": (lambda: load_secure("secure_resnet34_pretrained_pgd3_epoch_15.pth", 34),
                       dict(steps=10, alpha=lambda e: e / 4)),
}
EPS = [0.01, 0.03, 0.05, 0.07]
import os
ONLY = os.environ.get("ONLY")           # e.g. ONLY="Baseline CNN" re-runs just that model
res = json.load(open("../results/protocolA.json")) if ONLY else {}
for name, (loader_fn, cfg) in MODELS.items():
    if ONLY and name != ONLY:
        continue
    model, mean, std = loader_fn()
    ds = test_set(mean, std)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    r = {"clean": 0, "fgsm": {}, "pgd": {}}
    n = 0
    tot = {("fgsm", e): 0 for e in EPS} | {("pgd", e): 0 for e in EPS}
    t0 = time.time()
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        n += len(y)
        r["clean"] += accuracy(model, x, y)
        for e in EPS:
            tot[("fgsm", e)] += accuracy(model, fgsm_A(model, x, y, e), y)
            tot[("pgd", e)] += accuracy(model, pgd_A(model, x, y, e, cfg["alpha"](e), cfg["steps"],
                                                         rand_start=cfg.get("rand", True)), y)
    r["clean"] = 100 * r["clean"] / n
    for (k, e), v in tot.items():
        r[k][str(e)] = 100 * v / n
    res[name] = r
    print(name, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("../results/protocolA.json", "w"), indent=1)
