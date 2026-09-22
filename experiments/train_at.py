"""Pixel-space PGD adversarial training of plain / Secure ResNets (ablation study).

  python train_at.py --name r18_full --depth 18 --blur 1 --denoise 1 --avgmax 1

* inputs in [0,1]; mean/std normalisation happens inside the model; adversarial images clipped to [0,1]
* untargeted L-inf PGD, eps = 8/255, `steps` steps, step 2.5*eps/steps, random start
* pure adversarial loss (Madry et al.), SGD momentum 0.9, wd 5e-4, one-cycle LR, bf16 autocast
* 1,000 training images are held out as a validation set; the checkpoint with the best
  validation PGD-10 accuracy is saved as best (early stopping), the last epoch as final
Outputs: ../checkpoints/<name>_best.pt, <name>_final.pt and ../results/train_<name>.json
"""
import argparse, json, os, time, torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from common import *

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--depth", type=int, default=18)
p.add_argument("--blur", type=int, default=1)
p.add_argument("--denoise", type=int, default=1)
p.add_argument("--avgmax", type=int, default=1)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--act", default="relu", choices=["relu", "silu"])
p.add_argument("--ema", type=float, default=0.0, help="EMA decay for weight averaging (0 = off)")
p.add_argument("--loss", default="madry", choices=["madry", "trades"])
p.add_argument("--beta", type=float, default=6.0, help="TRADES KL weight")
p.add_argument("--quick", type=int, default=0, help="use only N training batches per epoch (speed test)")
a = p.parse_args()

torch.manual_seed(a.seed)
torch.backends.cudnn.benchmark = True
os.makedirs("../checkpoints", exist_ok=True)
EPS = 8 / 255

# data: 49,000 train / 1,000 validation split (fixed), test set is never touched here
tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
te_tf = T.ToTensor()
full_tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=tr_tf)
full_va = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=te_tf)
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(50000, generator=g).tolist()
tr_idx, va_idx = perm[1000:], perm[:1000]
tr_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_tr, tr_idx), batch_size=a.bs,
                                        shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
va_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_va, va_idx), batch_size=250, shuffle=False)

net = SecureNet(a.depth, bool(a.blur), bool(a.denoise), bool(a.avgmax), a.act)
model = Normalized(net, CIFAR_MEAN, CIFAR_STD).to(DEVICE).to(memory_format=torch.channels_last)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
ema = None
if a.ema > 0:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
n_it = a.epochs * (a.quick or len(tr_loader))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=n_it + 1, pct_start=0.3,
                                            anneal_strategy="linear", div_factor=25, final_div_factor=1000)


def craft(x, y, steps, alpha):
    adv = torch.clamp(x + torch.empty_like(x).uniform_(-EPS, EPS), 0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(adv), y)
        g_ = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g_.sign()
        adv = torch.clamp(x + torch.clamp(adv - x, -EPS, EPS), 0, 1).detach()
    return adv


def craft_trades(x, steps, alpha):
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    adv = torch.clamp(x + 0.001 * torch.randn_like(x), 0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.kl_div(F.log_softmax(model(adv), dim=1), p_nat, reduction="sum")
        g_ = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * g_.sign()
        adv = torch.clamp(x + torch.clamp(adv - x, -EPS, EPS), 0, 1).detach()
    model.train()
    return adv


def validate():
    m_eval = ema.module if ema is not None else model
    m_eval.eval(); c = r = n = 0
    for x, y in va_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            c += (m_eval(x).argmax(1) == y).sum().item()
        xa = pgd_B(m_eval, x, y, EPS, steps=10, alpha=2 / 255)
        with torch.no_grad():
            r += (m_eval(xa).argmax(1) == y).sum().item()
        n += len(y)
    model.train()
    return 100 * c / n, 100 * r / n


log, best = [], -1
t0 = time.time()
STATE = f"../checkpoints/{a.name}_state.pt"
start_ep = 1
if os.path.exists(STATE):                      # resume after an interruption
    st = torch.load(STATE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    if ema is not None and st.get("ema") is not None:
        ema.load_state_dict(st["ema"])
    log, best, start_ep = st["log"], st["best"], st["epoch"] + 1
    t0 = time.time() - 60 * (log[-1]["minutes"] if log else 0)
    print("resumed from epoch", st["epoch"], flush=True)
for ep in range(start_ep, a.epochs + 1):
    model.train(); tl = ta = tn = 0
    for i, (x, y) in enumerate(tr_loader):
        if a.quick and i >= a.quick:
            break
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        if a.loss == "madry":
            xa = craft(x, y, a.steps, 2.5 * EPS / a.steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xa); loss = F.cross_entropy(out, y)
        else:  # TRADES: natural CE + beta * KL(natural || adversarial)
            xa = craft_trades(x, a.steps, 2.5 * EPS / a.steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_nat = model(x); out = model(xa)
                loss = F.cross_entropy(out_nat, y) + a.beta * F.kl_div(
                    F.log_softmax(out.float(), 1), F.softmax(out_nat.float(), 1), reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if ema is not None:
            ema.update_parameters(model)
        tl += loss.item() * len(y); ta += (out.argmax(1) == y).sum().item(); tn += len(y)
    vc, vr = validate()
    rec = dict(epoch=ep, train_loss=tl / tn, train_adv_acc=100 * ta / tn, val_clean=vc, val_pgd10=vr,
               lr=sched.get_last_lr()[0], minutes=(time.time() - t0) / 60)
    log.append(rec); print(json.dumps(rec), flush=True)
    if vr > best:
        best = vr; torch.save((ema.module.model if ema is not None else net).state_dict(), f"../checkpoints/{a.name}_best.pt")
    json.dump(dict(args=vars(a), log=log, best_val_pgd10=best), open(f"../results/train_{a.name}.json", "w"), indent=1)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    ema=ema.state_dict() if ema is not None else None, log=log, best=best, epoch=ep), STATE)
torch.save((ema.module.model if ema is not None else net).state_dict(), f"../checkpoints/{a.name}_final.pt")
print("done", a.name, "best val PGD-10:", best)
