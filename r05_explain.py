"""
r05_explain.py - Feature importance of the main model (RF, core feature set).
  * Permutation importance computed OUT-OF-FOLD: for each outer fold of each
    repetition, the RF tuned on the outer-training data is evaluated on the
    held-out fold, and the drop in AUROC after permuting each feature is recorded.
  * SHAP (TreeExplainer) on a RF refitted on all patients with the median of the
    hyper-parameters selected in the nested CV (descriptive only).
Figures 3 and 6.
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from common import load_clean, OOF_DIR, TAB_DIR, FIG_DIR, CORE, LABELS, SEEDS, N_OUTER

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
df = load_clean()
res = pickle.load(open(os.path.join(OOF_DIR, "nested_results.pkl"), "rb"))
X = df[CORE]
names = [LABELS[c] for c in CORE]
out = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
shap_store = {}
for j, (outcome, title) in enumerate([("COMPLICATED", "Complicated appendicitis"), ("ABSCESS", "Intraoperative abscess")]):
    y = df[outcome].values
    imps = []
    hp = []
    for r in res:
        if r["outcome"] == outcome and r["fs"] == "core" and r["model"] == "Random forest":
            bp = {k.replace("clf__", ""): v for k, v in r["best_params"].items()}
            hp.append(bp)
            outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=r["seed"])
            tr, te = list(outer.split(X, y))[r["fold"]]
            rf = RandomForestClassifier(class_weight="balanced", random_state=r["seed"], n_jobs=-1, **bp)
            rf.fit(X.iloc[tr], y[tr])
            pi = permutation_importance(rf, X.iloc[te], y[te], scoring="roc_auc", n_repeats=20,
                                        random_state=r["seed"], n_jobs=-1)
            imps.append(pi.importances_mean)
    imps = np.array(imps)
    m, sd = imps.mean(0), imps.std(0, ddof=1)
    order = np.argsort(m)
    ax = axes[j]
    ax.barh(np.array(names)[order], m[order], xerr=sd[order], color="#1f4e8c", alpha=.85, capsize=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Mean decrease in out-of-fold AUROC (+/- SD over 25 folds)")
    ax.set_title(f"{'AB'[j]}  {title}", loc="left", fontsize=10)
    out[outcome] = {"permutation": {names[i]: [float(m[i]), float(sd[i])] for i in range(len(names))}}
    hpdf = pd.DataFrame(hp)
    final_hp = dict(n_estimators=int(hpdf.n_estimators.median()), max_depth=int(hpdf.max_depth.median()),
                    min_samples_leaf=int(hpdf.min_samples_leaf.median()))
    out[outcome]["final_hp"] = final_hp
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **final_hp).fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    shap_store[outcome] = sv
    out[outcome]["shap_mean_abs"] = {names[i]: float(np.abs(sv[:, i]).mean()) for i in range(len(names))}
plt.suptitle("Permutation importance of the main random-forest model (held-out outer folds of the repeated nested CV)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure3.png"), dpi=300); plt.close()

fig = plt.figure(figsize=(11, 4.2))
for j, (outcome, title) in enumerate([("COMPLICATED", "Complicated appendicitis"), ("ABSCESS", "Intraoperative abscess")]):
    plt.subplot(1, 2, j + 1)
    shap.summary_plot(shap_store[outcome], X.values, feature_names=names, show=False, plot_size=None, color_bar=(j == 1))
    plt.title(f"{'AB'[j]}  {title}", loc="left", fontsize=10)
    plt.xlabel("SHAP value (impact on predicted risk)", fontsize=8.5)
plt.suptitle("SHAP summary of the main random-forest model refitted on all patients (descriptive)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure6.png"), dpi=300); plt.close()
json.dump(out, open(os.path.join(TAB_DIR, "importance.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
