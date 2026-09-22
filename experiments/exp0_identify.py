"""Clean test accuracy of every checkpoint -> results/checkpoint_clean_acc.csv"""
import csv, glob, os, time
import torch
from common import *

rows = []
loader = pixel_loader(batch=250)
for path in sorted(glob.glob(os.path.join(PROJECT, "*.pth"))):
    name = os.path.basename(path)
    depth = 34 if "resnet34" in name else 18
    try:
        if name.startswith("baseline"):
            mm = load_baseline(name)
        else:
            mm = load_secure(name, depth)
    except Exception as e:  # earlier custom SecureCNN prototypes use other layers
        rows.append([name, "n/a", "architecture differs: " + type(e).__name__])
        print(name, "SKIP", type(e).__name__)
        continue
    m = wrap01(mm)
    c = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (m(x).argmax(1) == y).sum().item(); n += len(y)
    rows.append([name, f"{100*c/n:.2f}", ""])
    print(f"{name:50s} {100*c/n:6.2f}")

os.makedirs("../results", exist_ok=True)
with open("../results/checkpoint_clean_acc.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["checkpoint", "clean_acc_pct", "note"]); w.writerows(rows)
