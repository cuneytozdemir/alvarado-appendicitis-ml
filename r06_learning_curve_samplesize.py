"""
r06_learning_curve_samplesize.py - Learning curve and minimum sample size (R2, R3-5, R4).

Learning curve: random forest (core feature set, median nested-CV hyper-parameters)
trained on 20%-100% of each training split of a 5-fold CV repeated 10 times; the
AUROC is always evaluated on the full held-out fold.

Minimum sample size: Riley et al. (BMJ 2020) criteria for a binary-outcome
prediction model, assuming a conservative Cox-Snell R2 = 15% of its maximum.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold

from common import load_clean, TAB_DIR, FIG_DIR, CORE

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
df = load_clean()
imp = json.load(open(os.path.join(TAB_DIR, "importance.json")))
out = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fracs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
for j, (outcome, title) in enumerate([("COMPLICATED", "Complicated appendicitis"), ("ABSCESS", "Intraoperative abscess")]):
    y = df[outcome].values
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1, **imp[outcome]["final_hp"])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    sizes, tr_s, te_s = learning_curve(rf, df[CORE], y, train_sizes=fracs, cv=cv, scoring="roc_auc",
                                       n_jobs=15, shuffle=True, random_state=42)
    out[outcome] = dict(n_train=sizes.tolist(), test_mean=te_s.mean(1).tolist(), test_sd=te_s.std(1).tolist(),
                        train_mean=tr_s.mean(1).tolist())
    ax = axes[j]
    ax.errorbar(sizes, te_s.mean(1), yerr=te_s.std(1), marker="o", color="#1f4e8c", capsize=3, label="Held-out folds")
    ax.plot(sizes, tr_s.mean(1), marker="s", color="0.6", label="Training data")
    ax.set_ylim(0.5, 1.0); ax.set_xlabel("Number of training patients"); ax.set_ylabel("AUROC")
    ax.set_title(f"{'AB'[j]}  {title}", loc="left", fontsize=10); ax.legend(fontsize=8)
plt.suptitle("Learning curves of the main random-forest model (5-fold CV repeated 10 times; mean +/- SD)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figureS1_learning_curve.png"), dpi=300); plt.close()


def riley(prev, n_params, shrink=0.9, r2_frac=0.15):
    lnL0 = prev * np.log(prev) + (1 - prev) * np.log(1 - prev)
    max_r2 = 1 - np.exp(2 * lnL0)
    r2 = r2_frac * max_r2
    n1 = n_params / ((shrink - 1) * np.log(1 - r2 / shrink))
    s_req = r2 / (r2 + 0.05 * max_r2)                                       # criterion 2
    n2 = n_params / ((s_req - 1) * np.log(1 - r2 / s_req))
    n3 = (1.96 / 0.05) ** 2 * prev * (1 - prev)
    return dict(max_r2=max_r2, r2_cs=r2, n_criterion1=n1, n_criterion2=n2, n_criterion3=n3,
                n_required=float(np.ceil(max(n1, n2, n3))))


ss = {}
for outcome in ["COMPLICATED", "ABSCESS"]:
    prev = df[outcome].mean()
    ev = int(df[outcome].sum())
    # parameters: Alvarado group treated as 1 ordinal parameter
    for lab, k in [(f"core ({len(CORE)} parameters)", len(CORE))]:
        ss[f"{outcome}|{lab}"] = dict(events=ev, EPV=ev / k, **riley(prev, k))
out["sample_size"] = ss
json.dump(out, open(os.path.join(TAB_DIR, "learning_curve_samplesize.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
