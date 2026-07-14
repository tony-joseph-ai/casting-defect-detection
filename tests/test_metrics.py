"""Tests for src/metrics.py.  Run:  PYTHONPATH=. python tests/test_metrics.py"""
import numpy as np
from src.metrics import binary_metrics, roc_auc, sweep_thresholds, confusion

rng = np.random.default_rng(0)

def brute_auc(y, p):
    pos, neg = p[y==1], p[y==0]
    wins = sum((a > b) + 0.5*(a == b) for a in pos for b in neg)
    return wins / (len(pos)*len(neg))

# 1. AUC vs brute force, including heavy ties
for trial in range(5):
    y = rng.integers(0, 2, 60)
    p = np.round(rng.random(60), 1)          # forces ties
    assert abs(roc_auc(y, p) - brute_auc(y, p)) < 1e-9, "AUC mismatch"
print("AUC matches brute-force (with ties):        OK")

# 2. Perfect separation -> AUC 1.0, recall 1.0
y = np.array([0,0,0,1,1,1]); p = np.array([.1,.2,.3,.7,.8,.9])
m = binary_metrics(y, p, 0.5)
assert m["roc_auc"] == 1.0 and m["recall_defect"] == 1.0 and m["fn"] == 0
print("Perfect separation -> AUC=1, FN=0:          OK")

# 3. The lazy "always OK" model: high accuracy, ZERO defect recall.
#    This is exactly the failure mode the project is built to expose.
y = np.array([0]*90 + [1]*10); p = np.zeros(100)
m = binary_metrics(y, p, 0.5)
assert m["accuracy"] == 0.90 and m["recall_defect"] == 0.0 and m["fn"] == 10
print(f"'Always OK' model: acc={m['accuracy']:.2f} but recall={m['recall_defect']:.2f}, "
      f"{m['fn']} defects shipped:  OK")

# 4. Threshold sweep must lower the threshold when FN is 10x costlier than FP
y = rng.integers(0, 2, 300)
p = np.clip(0.5 + 0.25*(y*2-1) + rng.normal(0, .2, 300), .01, .99)
_, _, best_bal  = sweep_thresholds(y, p, cost_fn=1.0,  cost_fp=1.0)
_, _, best_cost = sweep_thresholds(y, p, cost_fn=10.0, cost_fp=1.0)
assert best_cost < best_bal, "cost-sensitive threshold should be LOWER"
print(f"Cost-sensitive threshold {best_cost:.2f} < balanced {best_bal:.2f}: OK")

# 5. Confusion matrix orientation
tn, fp, fn, tp = confusion(np.array([0,0,1,1]), np.array([0,1,0,1]))
assert (tn, fp, fn, tp) == (1,1,1,1)
print("Confusion matrix orientation:               OK")
print("\nAll metric tests passed.")
