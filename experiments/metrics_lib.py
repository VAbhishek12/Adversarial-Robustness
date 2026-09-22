"""Classification and robustness metrics used in the paper and the thesis.

Classification (computed from predictions, clean or adversarial):
  accuracy, balanced accuracy, macro / weighted precision, recall, F1, per-class P/R/F1,
  Matthews correlation coefficient (MCC), Cohen's kappa, macro one-vs-rest AUROC,
  mean cross-entropy (NLL), expected calibration error (ECE, 15 bins), mean confidence,
  mean confidence of wrong predictions (over-confidence).
Robustness:
  robust accuracy, accuracy drop, robustness ratio (adv/clean accuracy), attack success rate
  (fraction of correctly classified samples that the attack flips), robustness score
  (mean robust accuracy over an epsilon grid), critical epsilon (smallest L-inf budget, on a grid,
  at which PGD flips a correctly classified sample), input-gradient norm.
"""
import numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import (precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
                             roc_auc_score, balanced_accuracy_score, confusion_matrix)
from common import pgd_B


def ece_score(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def classification_metrics(prob, y):
    """prob: (N,10) softmax outputs (numpy), y: (N,) labels (numpy)."""
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(10), zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, labels=range(10), average="weighted", zero_division=0)
    try:
        auc = float(roc_auc_score(y, prob, multi_class="ovr", labels=list(range(10))))
    except ValueError:
        auc = float("nan")
    nll = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())
    wrong = pred != y
    return dict(
        accuracy=100 * float((pred == y).mean()), balanced_accuracy=100 * float(balanced_accuracy_score(y, pred)),
        precision_macro=100 * float(p.mean()), recall_macro=100 * float(r.mean()), f1_macro=100 * float(f.mean()),
        precision_weighted=100 * float(pw), recall_weighted=100 * float(rw), f1_weighted=100 * float(fw),
        mcc=float(matthews_corrcoef(y, pred)), kappa=float(cohen_kappa_score(y, pred)), auroc_macro=auc,
        nll=nll, ece=100 * ece_score(prob, y), mean_confidence=100 * float(prob.max(1).mean()),
        confidence_when_wrong=100 * float(prob.max(1)[wrong].mean()) if wrong.any() else float("nan"),
        per_class_precision=(100 * p).tolist(), per_class_recall=(100 * r).tolist(), per_class_f1=(100 * f).tolist(),
        confusion=confusion_matrix(y, pred, labels=range(10)).tolist())


@torch.no_grad()
def probs(model, x):
    return F.softmax(model(x), 1).float().cpu().numpy()


def collect(model, loader, attack=None, device="cuda"):
    """Runs the model (optionally on adversarial versions of the images); returns (prob, y, x_correct_mask)."""
    P, Y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack is not None:
            x = attack(model, x, y)
        P.append(probs(model, x)); Y.append(y.cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def attack_success_rate(prob_clean, prob_adv, y):
    ok = prob_clean.argmax(1) == y
    return 100 * float((prob_adv.argmax(1)[ok] != y[ok]).mean()) if ok.any() else float("nan")


def critical_epsilon(model, loader, grid255=(0.5, 1, 2, 3, 4, 6, 8, 12, 16), steps=10, device="cuda"):
    """Smallest L-inf budget on the grid at which PGD flips a correctly classified sample (in 1/255 units).
    Samples never flipped within the grid are assigned 2*max(grid) (censored)."""
    crit = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            ok = model(x).argmax(1) == y
        c = torch.full((len(y),), float(2 * max(grid255)), device=device)
        found = torch.zeros(len(y), dtype=torch.bool, device=device)
        for e in grid255:
            xa = pgd_B(model, x, y, e / 255, steps)
            with torch.no_grad():
                fooled = (model(xa).argmax(1) != y) & ~found
            c[fooled] = e; found |= fooled
        crit.append(c[ok].cpu().numpy())
    crit = np.concatenate(crit)
    return dict(mean=float(crit.mean()), median=float(np.median(crit)))


def input_gradient_norm(model, loader, device="cuda"):
    g = []
    for x, y in loader:
        x, y = x.to(device).requires_grad_(True), y.to(device)
        gx = torch.autograd.grad(F.cross_entropy(model(x), y, reduction="sum"), x)[0]
        g.append(gx.abs().flatten(1).sum(1).cpu())
    g = torch.cat(g)
    return dict(mean=float(g.mean()), median=float(g.median()))
