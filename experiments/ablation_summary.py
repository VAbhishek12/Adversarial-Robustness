"""Prints the key numbers and deltas of the retrained models (used to write the results text quickly)."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = json.load(open(os.path.join(R, "new_eval_partial.json"))); E = json.load(open(os.path.join(R, "new_eval.json"))) if os.path.exists(os.path.join(R, "new_eval.json")) else {}
S = {**P, **E}; M = json.load(open(os.path.join(R, "metrics.json")))
ref = S["r18_plain"]
print(f"{'model':16s} {'clean':>6} {'PGD8_10k':>8} {'AA8':>6} {'sw2':>6} {'sw4':>6} {'sw8':>6} {'sw12':>6} {'sw16':>6} | dClean dPGD8 | F1c F1a8 ASR8 crit  | final8  params")
for n, r in S.items():
    sw = r.get("pgd20_sweep_2000", {}); m = M.get(n)
    line = f"{n:16s} {r.get('clean',float('nan')):6.2f} {r.get('pgd20_8_full',float('nan')):8.2f} {r.get('autoattack_8',float('nan')):6.1f} " + " ".join(f"{sw.get(k,float('nan')):6.1f}" for k in ("2","4","8","12","16"))
    line += f" | {r.get('clean',0)-ref['clean']:+6.2f} {r.get('pgd20_8_full',0)-ref['pgd20_8_full']:+6.2f}"
    if m: line += f" | {m['clean']['f1_macro']:.1f} {m['adv']['8']['f1_macro']:.1f} {m['asr']['8']:.1f} {m['critical_eps']['mean']:.2f}"
    if "final" in r: line += f" | {r['final']['pgd20_8']:.1f} {r.get('params_M',float('nan')):.2f}"
    print(line)
