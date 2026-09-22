#!/usr/bin/env bash
# Full command sequence behind the thesis and paper results.
# Run from the repository root:  bash scripts/reproduce.sh
# Needs a CUDA GPU. Training takes about 30 to 80 minutes per network on an RTX 3050 (4 GB).
# Original checkpoints are expected in checkpoints/original (or set ORIG_CKPT_DIR).
set -euo pipefail
cd "$(dirname "$0")/../experiments"
export KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}
LOG=../results

# 1. original models: identification, first test (Protocol A), corrected test (Protocol B), cost, equal attack
$PY exp0_identify.py
$PY exp1_protocolA.py
$PY exp2_protocolB.py
$PY exp3_compute.py
$PY exp7_equal_attack.py

# 2. retrained networks, one shared recipe (pixel space PGD adversarial training at 8/255, 20 epochs, seed 0)
train() { name=$1; shift; $PY train_at.py --name "$name" "$@" > "$LOG/train_$name.log" 2>&1; }
train r18_plain     --depth 18 --blur 0 --denoise 0 --avgmax 0 --epochs 20
train r18_full      --depth 18 --blur 1 --denoise 1 --avgmax 1 --epochs 20
train r18_nodenoise --depth 18 --blur 1 --denoise 0 --avgmax 1 --epochs 20
train r18_noblur    --depth 18 --blur 0 --denoise 1 --avgmax 1 --epochs 20
train r18v2_plain   --depth 18 --blur 0 --denoise 0 --avgmax 0 --act silu --ema 0.995 --epochs 20

# 3. evaluation: clean, PGD-20 at 8/255, budget sweep, cost; AA=1 adds the APGD worst case (first 1000 images)
AA=1 $PY eval_new.py r18_plain r18_full r18v2_plain     > "$LOG/eval_group1.log" 2>&1
AA=0 $PY eval_new.py r18_nodenoise r18_noblur           > "$LOG/eval_group2.log" 2>&1
$PY eval_metrics.py                                      > "$LOG/eval_metrics.log" 2>&1

# 4. summary table
$PY summarize.py
echo "done"
