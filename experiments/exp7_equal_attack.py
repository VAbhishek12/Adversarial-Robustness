"""Equal-attack check of the paper's SecureCNN vs SecureResNet34 PGD comparison (Protocol A conventions).
The paper states PGD used a fixed step alpha=0.007 with 10 iterations. The SecureCNN numbers in the paper
match alpha=eps/4. Here BOTH secure models (and the baseline) are attacked with (a) alpha=0.007, 10 steps and
(b) alpha=eps/4, 10 steps, all with a random start, on all 10,000 test images.
Output: results/equal_attack.json"""
import json, torch
from common import *

torch.manual_seed(0)
FN = {"Baseline CNN": lambda: load_baseline(),
      "SecureCNN": lambda: load_secure("best_securecnn.pth", 18),
      "SecureResNet34": lambda: load_secure("secure_resnet34_pretrained_pgd3_epoch_15.pth", 34)}
EPS = [0.01, 0.03, 0.05, 0.07]
SETTINGS = {"alpha=0.007, 10 steps": lambda e: 0.007, "alpha=eps/4, 10 steps": lambda e: e / 4}
res = {}
for name, fn in FN.items():
    model, mean, std = fn()
    dl = torch.utils.data.DataLoader(test_set(mean, std), batch_size=128, shuffle=False)
    tot = {(s, e): 0. for s in SETTINGS for e in EPS}; n = 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE); n += len(y)
        for s, af in SETTINGS.items():
            for e in EPS:
                tot[(s, e)] += accuracy(model, pgd_A(model, x, y, e, af(e), 10), y)
    res[name] = {s: {str(e): 100 * tot[(s, e)] / n for e in EPS} for s in SETTINGS}
    print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open("../results/equal_attack.json", "w"), indent=1)
