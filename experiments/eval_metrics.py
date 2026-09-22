"""Full metric suite for every model (originals and retrained).  python eval_metrics.py [name ...]
Clean metrics: all 10,000 test images.  Adversarial metrics (PGD-20 at 2, 4, 8 /255 in pixel space,
random start): first 2,000 test images.  Robustness summaries: critical epsilon and gradient norm on the
first 1,000 images.  Output: results/metrics.json (updated incrementally)."""
import json, os, sys, time, torch
from common import *
from registry import CFG, OLD, build_new
from metrics_lib import *

torch.manual_seed(0)
OUT = "../results/metrics.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
EPS = (2, 4, 8)
N_ADV = 2000


def get(name):
    if name in OLD:
        return OLD[name]()
    return build_new(name, "best")


names = sys.argv[1:] or (list(OLD) + [n for n in CFG if os.path.exists(f"../checkpoints/{n}_final.pt")])
for name in names:
    if name in res:
        continue
    t0 = time.time(); m = get(name); r = {}
    Pc, Y = collect(m, pixel_loader(250), None, DEVICE)
    r["clean"] = classification_metrics(Pc, Y)
    Pc2, Y2 = Pc[:N_ADV], Y[:N_ADV]
    r["clean_subset_accuracy"] = 100 * float((Pc2.argmax(1) == Y2).mean())
    r["adv"], r["asr"] = {}, {}
    for e in EPS:
        Pa, Ya = collect(m, pixel_loader(125, N_ADV), lambda mm, x, y, e=e: pgd_B(mm, x, y, e / 255, 20), DEVICE)
        r["adv"][str(e)] = classification_metrics(Pa, Ya)
        r["asr"][str(e)] = attack_success_rate(Pc2, Pa, Ya)
    accs = [r["adv"][str(e)]["accuracy"] for e in EPS]
    r["robustness_score_mean_acc"] = float(sum(accs) / len(accs))
    r["robustness_ratio_8"] = r["adv"]["8"]["accuracy"] / r["clean_subset_accuracy"]
    r["accuracy_drop_8"] = r["clean_subset_accuracy"] - r["adv"]["8"]["accuracy"]
    r["critical_eps"] = critical_epsilon(m, pixel_loader(125, 1000), device=DEVICE)
    r["grad_norm_l1"] = input_gradient_norm(m, pixel_loader(125, 1000), DEVICE)
    res[name] = r
    json.dump(res, open(OUT, "w"), indent=1)
    print(name, "clean acc %.2f F1 %.2f | PGD8 acc %.2f F1 %.2f | ASR8 %.1f | crit-eps mean %.2f | %.1f min" % (
        r["clean"]["accuracy"], r["clean"]["f1_macro"], r["adv"]["8"]["accuracy"], r["adv"]["8"]["f1_macro"],
        r["asr"]["8"], r["critical_eps"]["mean"], (time.time() - t0) / 60), flush=True)
