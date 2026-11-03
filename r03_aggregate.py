"""
r03_aggregate.py - Combine repeated nested-CV predictions, inference, tables, ROC figure.

How repeated predictions are combined (R1-6, R2, R3-4, R4-6)
  For every model, each patient has exactly one OOF prediction per repetition
  (5 repetitions). For inference we average the 5 OOF probabilities per
  patient, so that every patient contributes exactly ONE prediction to the
  DeLong test, the patient-level bootstrap CI and the calibration/DCA
  analyses; this avoids treating repeated, correlated predictions of the same
  patient as independent observations. Repetition-to-repetition variability is
  reported separately (mean +/- SD and range of the per-repetition AUROCs and
  DeLong p-values).

Multiplicity (R1-5, R2, R3-4)
  Family A (primary, confirmatory): main model (random forest, core feature
     set) vs Alvarado risk group, for the two outcomes (complicated
     appendicitis, intraoperative abscess) -> Holm over m=2.
  Family B (exploratory, per outcome): the remaining nine algorithms and the
     averaging ensemble vs Alvarado -> Holm over m=10 within each outcome.
  Sensitivity analyses are reported with unadjusted p-values and labelled so.
"""
import os, json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

from common import load_clean, OOF_DIR, TAB_DIR, FIG_DIR, SEEDS, OUTCOMES, FEATURE_SETS
from stats_utils import delong_test, bootstrap_ci, bootstrap_diff_ci, holm

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
df = load_clean()
n = len(df)
res = pickle.load(open(os.path.join(OOF_DIR, "nested_results.pkl"), "rb"))

# ------------------------------------------------------------ assemble n x seeds matrices
P = defaultdict(lambda: np.full((n, len(SEEDS)), np.nan))
PC = defaultdict(lambda: np.full((n, len(SEEDS)), np.nan))
B = defaultdict(lambda: np.full((n, len(SEEDS)), np.nan))     # binary @ train-derived threshold
BC = defaultdict(lambda: np.full((n, len(SEEDS)), np.nan))
params = defaultdict(list)
from sklearn.model_selection import StratifiedKFold
from stats_utils import youden_threshold
for r in res:
    if r["model"] == "Alvarado risk group":   # recompute train-fold threshold on the ordinal group
        y_ = df[r["outcome"]].values
        tr, te = list(StratifiedKFold(5, shuffle=True, random_state=r["seed"]).split(df, y_))[r["fold"]]
        assert np.array_equal(te, r["te"])
        r["thr"] = youden_threshold(y_[tr], df["ALVARADO_GROUP"].values[tr])
res = [r for r in res if r["fs"] in FEATURE_SETS or r["fs"] == "alvarado"]
for r in res:
    k = (r["outcome"], r["fs"], r["model"])
    s = SEEDS.index(r["seed"])
    P[k][r["te"], s] = r["p"]
    B[k][r["te"], s] = (r["p"] >= r["thr"]).astype(float)
    if "p_cal" in r:
        PC[k][r["te"], s] = r["p_cal"]
        BC[k][r["te"], s] = (r["p_cal"] >= r["thr_cal"]).astype(float)
    if "best_params" in r:
        params[k].append(r["best_params"])
for k in list(P):
    assert not np.isnan(P[k]).any(), k

ENS = ["Random forest", "XGBoost", "LightGBM", "CatBoost"]
for outcome in ["COMPLICATED", "ABSCESS"]:
    P[(outcome, "core", "Ensemble (mean of RF, XGBoost, LightGBM, CatBoost)")] = \
        np.mean([P[(outcome, "core", m)] for m in ENS], axis=0)


def summarise(outcome, fs, model):
    y = df[outcome].values
    k = (outcome, fs, model)
    alv = P[(outcome, "alvarado", "Alvarado risk group")]
    pm = P[k].mean(1)
    per_seed = [roc_auc_score(y, P[k][:, s]) for s in range(len(SEEDS))]
    out = dict(outcome=outcome, feature_set=fs, model=model,
               AUROC=roc_auc_score(y, pm), AUROC_rep_mean=np.mean(per_seed), AUROC_rep_sd=np.std(per_seed, ddof=1),
               AUROC_rep_min=np.min(per_seed), AUROC_rep_max=np.max(per_seed),
               AUPRC=average_precision_score(y, pm))
    out["CI_low"], out["CI_high"] = bootstrap_ci(y, pm)
    if k in B:
        sens = [B[k][y == 1, s].mean() for s in range(len(SEEDS))]
        spec = [1 - B[k][y == 0, s].mean() for s in range(len(SEEDS))]
        out["Sensitivity"], out["Specificity"] = np.mean(sens), np.mean(spec)
    if model != "Alvarado risk group":
        _, _, p = delong_test(y, pm, alv.mean(1))
        out["p_DeLong_vs_Alvarado"] = p
        out["dAUROC"] = roc_auc_score(y, pm) - roc_auc_score(y, alv.mean(1))
        out["dAUROC_lo"], out["dAUROC_hi"] = bootstrap_diff_ci(y, pm, alv.mean(1))
        ps = [delong_test(y, P[k][:, s], alv[:, s])[2] for s in range(len(SEEDS))]
        out["p_rep_min"], out["p_rep_max"] = min(ps), max(ps)
        out["n_rep_sig"] = int(sum(p < 0.05 for p in ps))
    if not model.startswith(("Alvarado", "Ensemble")):
        out["Brier"] = brier_score_loss(y, pm)
    return out


rows = []
for outcome in OUTCOMES:
    for (o, fs, m) in list(P):
        if o == outcome:
            rows.append(summarise(o, fs, m))
T = pd.DataFrame(rows)

# ------------------------------------------------------------ multiplicity
T["p_Holm"] = np.nan
T["family"] = "sensitivity (unadjusted)"
prim = T[(T.feature_set == "core") & (T.model == "Random forest") & T.outcome.isin(["COMPLICATED", "ABSCESS"])]
adj = holm({i: T.loc[i, "p_DeLong_vs_Alvarado"] for i in prim.index})
for i, v in adj.items():
    T.loc[i, "p_Holm"] = v; T.loc[i, "family"] = "A: primary (Holm, m=2)"
for outcome in ["COMPLICATED", "ABSCESS"]:
    ex = T[(T.feature_set == "core") & (T.outcome == outcome) & (T.model != "Random forest") &
           (T.model != "Alvarado risk group")]
    adj = holm({i: T.loc[i, "p_DeLong_vs_Alvarado"] for i in ex.index})
    for i, v in adj.items():
        T.loc[i, "p_Holm"] = v; T.loc[i, "family"] = f"B: exploratory (Holm, m={len(adj)})"
T.loc[T.model == "Alvarado risk group", "family"] = "reference"
T = T.sort_values(["outcome", "feature_set", "AUROC"], ascending=[True, True, False])
T.to_csv(os.path.join(TAB_DIR, "all_models_performance.csv"), index=False)
pd.set_option("display.width", 250)
print(T[["outcome", "feature_set", "model", "AUROC", "CI_low", "CI_high", "AUROC_rep_mean", "AUROC_rep_sd",
         "AUPRC", "Sensitivity", "Specificity", "p_DeLong_vs_Alvarado", "p_Holm", "p_rep_min", "p_rep_max",
         "n_rep_sig"]].round(3).to_string())

# ------------------------------------------------------------ extra pairwise comparisons (sensitivity)
pairs = []
for outcome in ["COMPLICATED", "ABSCESS"]:
    y = df[outcome].values
    core = P[(outcome, "core", "Random forest")].mean(1)
    for fs in ["core_no_alvarado", "core_no_neutrophil"]:
        other = P[(outcome, fs, "Random forest")].mean(1)
        a, b, p = delong_test(y, core, other)
        pairs.append(dict(outcome=outcome, comparison=f"RF core vs RF {fs}", auc_a=a, auc_b=b, p=p))
    lr = P[(outcome, "core", "Logistic regression")].mean(1)
    a, b, p = delong_test(y, core, lr)
    pairs.append(dict(outcome=outcome, comparison="RF core vs LR core", auc_a=a, auc_b=b, p=p))
pairs = pd.DataFrame(pairs)
pairs.to_csv(os.path.join(TAB_DIR, "pairwise_sensitivity.csv"), index=False)
print(pairs.round(4).to_string())

# ------------------------------------------------------------ hyper-parameters selected
hp = []
for k, lst in params.items():
    if k[1] == "core":
        for d in lst:
            hp.append(dict(outcome=k[0], model=k[2], **{kk.replace("clf__", ""): v for kk, v in d.items()}))
pd.DataFrame(hp).to_csv(os.path.join(TAB_DIR, "selected_hyperparameters.csv"), index=False)

# ------------------------------------------------------------ save averaged predictions
avg = {f"{o}|{fs}|{m}": P[(o, fs, m)].mean(1) for (o, fs, m) in P}
avg.update({f"CAL|{o}|{fs}|{m}": PC[(o, fs, m)].mean(1) for (o, fs, m) in PC})
pickle.dump(dict(avg=avg, PC={str(k): v for k, v in PC.items()}, BC={str(k): v for k, v in BC.items()}),
            open(os.path.join(OOF_DIR, "averaged_predictions.pkl"), "wb"))

# ------------------------------------------------------------ Figure 4: ROC curves (core set)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, outcome, title in [(axes[0], "COMPLICATED", "A  Complicated appendicitis (grade 3-4)"),
                           (axes[1], "ABSCESS", "B  Intraoperative abscess")]:
    y = df[outcome].values
    sub = T[(T.outcome == outcome) & (T.feature_set.isin(["core", "alvarado"]))]
    for _, r in sub.iterrows():
        key = (outcome, r.feature_set, r.model)
        p = P[key].mean(1)
        fpr, tpr, _ = roc_curve(y, p)
        if r.model == "Random forest":
            ax.plot(fpr, tpr, color="#1f4e8c", lw=2.6, label=f"Random forest, main model (AUROC {r.AUROC:.3f})", zorder=5)
        elif r.model == "Alvarado risk group":
            ax.plot(fpr, tpr, color="#d9822b", lw=2.6, label=f"Alvarado risk group (AUROC {r.AUROC:.3f})", zorder=4)
        else:
            ax.plot(fpr, tpr, color="0.72", lw=0.9, zorder=1)
    ax.plot([], [], color="0.72", lw=0.9, label="Other algorithms (n=10; see Additional file 1)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=7.6); ax.set_aspect("equal")
fig.suptitle("Out-of-fold ROC curves from 5-fold nested cross-validation repeated with 5 seeds\n"
             "(predictors: age, sex, leukocyte count, CRP, Alvarado risk group; one averaged prediction per patient)",
             fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure4.png"), dpi=300); plt.close()
print("[r03] done")
