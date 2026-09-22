"""Evaluation of the retrained ablation models (pixel space, eps in 1/255 units).
For every model whose checkpoint exists and is not yet in results/new_eval.json:
  * clean accuracy, all 10,000 test images
  * PGD-20 at 8/255 on all 10,000 images; PGD-20 at 2,4,12,16/255 on the first 2,000
  * APGD worst case (APGD-CE and APGD-DLR) at 8/255 and 4/255 on the first 1,000 (500 for 4/255)
  * final-epoch checkpoint: clean and PGD-20 at 8/255 on the first 2,000 (robust-overfitting check)
  * params / MACs / latency
Output: results/new_eval.json (updated incrementally)
"""
import json, os, sys, time, torch, torchattacks
from torch.utils.flop_counter import FlopCounterMode
from common import *

torch.manual_seed(0)
from registry import CFG, build_new
OUT = "../results/new_eval.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def build(name, which):
    return build_new(name, which)


def acc_pgd(m, eps255, n=None, steps=20):
    ok = tot = 0
    for x, y in pixel_loader(125, n):
        x, y = x.to(DEVICE), y.to(DEVICE)
        ok += accuracy(m, pgd_B(m, x, y, eps255 / 255, steps), y); tot += len(y)
    return 100 * ok / tot


def acc_clean(m, n=None):
    ok = tot = 0
    for x, y in pixel_loader(250, n):
        x, y = x.to(DEVICE), y.to(DEVICE)
        ok += accuracy(m, x, y); tot += len(y)
    return 100 * ok / tot


def acc_aa(m, eps255, n):
    """APGD worst case: APGD with the cross entropy loss and APGD with the DLR loss (100 steps, one restart each).
    An image counts as robust only if it survives both. This is the same test as the 'worst case' of Protocol B."""
    ok = tot = 0
    a1 = torchattacks.APGD(m, norm="Linf", eps=eps255 / 255, steps=100, n_restarts=1, loss="ce", seed=0)
    a2 = torchattacks.APGD(m, norm="Linf", eps=eps255 / 255, steps=100, n_restarts=1, loss="dlr", seed=0)
    for x, y in pixel_loader(250, n):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            c1 = m(a1(x, y)).argmax(1) == y
            c2 = m(a2(x, y)).argmax(1) == y
        ok += (c1 & c2).sum().item(); tot += len(y)
    return 100 * ok / tot


names = [n for n in CFG if os.path.exists(f"../checkpoints/{n}_best.pt") and n not in res] if len(sys.argv) < 2 else sys.argv[1:]
PART = "../results/new_eval_partial.json"          # stage results are saved as they finish, so a run can resume
part = json.load(open(PART)) if os.path.exists(PART) else {}
for name in names:
    if not os.path.exists(f"../checkpoints/{name}_final.pt"):
        print("skip (training not finished):", name); continue
    t0 = time.time(); m = build(name, "best"); r = part.get(name, {})
    def stage(key, fn):
        if key not in r:
            r[key] = fn()
            cur = json.load(open(PART)) if os.path.exists(PART) else {}
            cur[name] = r; json.dump(cur, open(PART, "w"), indent=1); print(name, key, r[key], flush=True)
    stage("clean", lambda: acc_clean(m))
    stage("pgd20_8_full", lambda: acc_pgd(m, 8))
    stage("pgd20_sweep_2000", lambda: {str(e): acc_pgd(m, e, 2000) for e in (2, 4, 8, 12, 16)})
    if os.environ.get("AA", "1") == "1":
        stage("autoattack_8", lambda: acc_aa(m, 8, 1000))
    mf = build(name, "final")
    stage("final", lambda: dict(clean=acc_clean(mf, 2000), pgd20_8=acc_pgd(mf, 8, 2000)))
    params = sum(p.numel() for p in m.parameters())
    x1 = torch.rand(1, 3, 32, 32, device=DEVICE)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x1)
    with torch.no_grad():
        for _ in range(30): m(x1)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200): m(x1)
        torch.cuda.synchronize(); lat = (time.perf_counter() - t) / 200 * 1000
    r["params_M"] = params / 1e6; r["MACs_M"] = fc.get_total_flops() / 2e6; r["latency_ms"] = lat
    r.setdefault("autoattack_4", float("nan"))          # 4/255 AutoAttack dropped to fit the compute budget
    curr = json.load(open(OUT)) if os.path.exists(OUT) else {}
    curr[name] = r; res = curr
    json.dump(curr, open(OUT, "w"), indent=1)
    print(name, "DONE", json.dumps(r), f"{(time.time()-t0)/60:.1f} min", flush=True)

# transfer among the finished models (PGD-20 at 8/255, first 2,000 images)
done = [n for n in CFG if n in res]
if len(done) >= 2 and os.environ.get("TRANSFER") == "1":
    T_ = {}
    ms = {n: build(n, "best") for n in done}
    for src in done:
        for dst in done:
            ok = tot = 0
            for x, y in pixel_loader(125, 2000):
                x, y = x.to(DEVICE), y.to(DEVICE)
                xa = pgd_B(ms[src], x, y, 8 / 255, 20)
                ok += accuracy(ms[dst], xa, y); tot += len(y)
            T_[f"{src}->{dst}"] = 100 * ok / tot
    res["_meta"] = {"transfer": T_, "models": done}
    json.dump(res, open(OUT, "w"), indent=1)
    print("transfer done")
