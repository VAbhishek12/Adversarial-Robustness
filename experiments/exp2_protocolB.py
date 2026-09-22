"""Protocol B: corrected evaluation in pixel space.
Inputs are [0,1] pixels, normalisation happens inside the model, adversarial
images are clamped to [0,1], eps in {1,2,4,8,16}/255.
 - FGSM and PGD-20 (random start, alpha=2.5*eps/20) on the full 10,000 test images
 - APGD-CE and APGD-DLR (torchattacks, 100 steps) on the first 1,000 test images
Output: results/protocolB.json
"""
import json, time, torch, torchattacks
from common import *

torch.manual_seed(0)
MODELS = {
    "Baseline CNN":   lambda: load_baseline(),
    "SecureCNN":      lambda: load_secure("best_securecnn.pth", 18),
    "SecureResNet34": lambda: load_secure("secure_resnet34_pretrained_pgd3_epoch_15.pth", 34),
}
EPS = [1, 2, 4, 8, 16]
N_STRONG = 1000
res = {}
for name, fn in MODELS.items():
    model = wrap01(fn())
    r = {"clean": 0., "fgsm": {}, "pgd20": {}, "apgd_ce": {}, "apgd_dlr": {}, "worst": {}}
    n = 0
    tot = {(k, e): 0. for k in ("fgsm", "pgd20") for e in EPS}
    t0 = time.time()
    for x, y in pixel_loader(128):
        x, y = x.to(DEVICE), y.to(DEVICE)
        n += len(y)
        r["clean"] += accuracy(model, x, y)
        for e in EPS:
            eps = e / 255
            tot[("fgsm", e)] += accuracy(model, fgsm_B(model, x, y, eps), y)
            tot[("pgd20", e)] += accuracy(model, pgd_B(model, x, y, eps, 20), y)
    r["clean"] = 100 * r["clean"] / n
    for (k, e), v in tot.items():
        r[k][str(e)] = 100 * v / n
    # stronger attacks on a subset
    for e in EPS:
        eps = e / 255
        ok_ce = ok_dlr = ok_both = m = 0
        for x, y in pixel_loader(125, N_STRONG):
            x, y = x.to(DEVICE), y.to(DEVICE)
            a1 = torchattacks.APGD(model, norm="Linf", eps=eps, steps=100, n_restarts=1, loss="ce", seed=0)
            a2 = torchattacks.APGD(model, norm="Linf", eps=eps, steps=100, n_restarts=1, loss="dlr", seed=0)
            with torch.no_grad():
                c1 = model(a1(x, y)).argmax(1) == y
                c2 = model(a2(x, y)).argmax(1) == y
            ok_ce += c1.sum().item(); ok_dlr += c2.sum().item()
            ok_both += (c1 & c2).sum().item(); m += len(y)
        r["apgd_ce"][str(e)] = 100 * ok_ce / m
        r["apgd_dlr"][str(e)] = 100 * ok_dlr / m
        r["worst"][str(e)] = 100 * ok_both / m
    res[name] = r
    print(name, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("../results/protocolB.json", "w"), indent=1)
