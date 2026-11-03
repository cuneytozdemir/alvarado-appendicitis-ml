"""
r01_descriptive.py - Baseline table, Alvarado group x histopathology association,
odds ratios, univariate discrimination, correlation matrix, Figures 1-2.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import load_clean, FIG_DIR, TAB_DIR, LABELS
from stats_utils import delong_ci, logit_fit

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
df = load_clean()
out = {"cleaning_log": df.attrs["cleaning_log"], "n": len(df)}


def med_iqr(x):
    q = np.percentile(x, [50, 25, 75])
    return f"{q[0]:,.0f} [{q[1]:,.0f}-{q[2]:,.0f}]"


# ------------------------------------------------------------ Table 2
rows = []
s, c = df[df.COMPLICATED == 0], df[df.COMPLICATED == 1]
for var, lab in [("AGE", "Age (years), median [IQR]"), ("WBC", "Leukocyte (/uL)"), ("CRP", "CRP (mg/L)")]:
    p = stats.mannwhitneyu(s[var], c[var]).pvalue
    f = med_iqr
    rows.append([lab, f(s[var]), f(c[var]), p])
tab = pd.crosstab(df.MALE, df.COMPLICATED)
rows.insert(1, ["Male, n (%)", f"{s.MALE.sum()} ({s.MALE.mean()*100:.1f})",
                f"{c.MALE.sum()} ({c.MALE.mean()*100:.1f})", stats.chi2_contingency(tab)[1]])
for g, lab in [(1, "Alvarado group 1 (score 5-6), n (%)"), (2, "Alvarado group 2 (score 7-8), n (%)"),
               (3, "Alvarado group 3 (score 9-10), n (%)")]:
    rows.append([lab, f"{(s.ALVARADO_GROUP==g).sum()} ({(s.ALVARADO_GROUP==g).mean()*100:.1f})",
                 f"{(c.ALVARADO_GROUP==g).sum()} ({(c.ALVARADO_GROUP==g).mean()*100:.1f})",
                 stats.chi2_contingency(pd.crosstab(df.ALVARADO_GROUP, df.COMPLICATED))[1] if g == 1 else np.nan])
rows.append(["Intraoperative abscess, n (%)", f"{s.ABSCESS.sum()} ({s.ABSCESS.mean()*100:.1f})",
             f"{c.ABSCESS.sum()} ({c.ABSCESS.mean()*100:.1f})",
             stats.chi2_contingency(pd.crosstab(df.ABSCESS, df.COMPLICATED))[1]])
t2 = pd.DataFrame(rows, columns=["Variable", "Simple", "Complicated", "p"])
t2.to_csv(os.path.join(TAB_DIR, "table2_baseline.csv"), index=False)
print(t2.to_string())

out["sample"] = dict(age_median=float(df.AGE.median()), age_iqr=list(np.percentile(df.AGE, [25, 75])),
                     age_range=[int(df.AGE.min()), int(df.AGE.max())], male=int(df.MALE.sum()),
                     wbc_median=float(df.WBC.median()),
                     crp_median=float(df.CRP.median()), crp_iqr=list(np.percentile(df.CRP, [25, 75])),
                     laparoscopic=int(df.LAPAROSCOPIC.sum()), los_median=float(df.LOS.median()),
                     los_range=[float(df.LOS.min()), float(df.LOS.max())],
                     complicated=int(df.COMPLICATED.sum()), abscess=int(df.ABSCESS.sum()),
                     composite=int(df.COMPOSITE.sum()))

# ------------------------------------------------------------ Alvarado x histopathology
ct = pd.crosstab(df.ALVARADO_GROUP, df.HISTO)
ct.to_csv(os.path.join(TAB_DIR, "table3_alvarado_histo.csv"))
chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
rho, prho = stats.spearmanr(df.ALVARADO_GROUP, df.HISTO)
# Cochran-Armitage trend test (complicated proportion across ordered groups)
g = pd.crosstab(df.ALVARADO_GROUP, df.COMPLICATED)
w = np.array(g.index, float); r = g[1].values; nn = g.sum(1).values
pbar = r.sum() / nn.sum()
T = np.sum(w * (r - nn * pbar))
VarT = pbar * (1 - pbar) * (np.sum(nn * w**2) - np.sum(nn * w)**2 / nn.sum())
z = T / np.sqrt(VarT); p_ca = 2 * stats.norm.sf(abs(z))
D = pd.get_dummies(df.ALVARADO_GROUP, prefix="G", drop_first=True).astype(float)
ors = logit_fit(D.values, df.COMPLICATED, ["G_2", "G_3"])
ors_abs = logit_fit(D.values, df.ABSCESS, ["G_2", "G_3"])
out["alvarado_histo"] = dict(chi2=chi2, p=p, dof=dof, cramers_v=V, spearman=rho, p_spearman=prho,
                             cochran_armitage_p=p_ca, or_complicated=ors, or_abscess=ors_abs,
                             complicated_rate_by_group={int(k): float(v) for k, v in
                                                        df.groupby("ALVARADO_GROUP").COMPLICATED.mean().items()},
                             abscess_rate_by_group={int(k): float(v) for k, v in
                                                    df.groupby("ALVARADO_GROUP").ABSCESS.mean().items()})
ct_ab = pd.crosstab(df.ABSCESS, df.HISTO)
ct_ab.to_csv(os.path.join(TAB_DIR, "table7_abscess_histo.csv"))
out["abscess_histo"] = ct_ab.to_dict()

# ------------------------------------------------------------ univariate discrimination
uni = []
for v in ["ALVARADO_GROUP", "WBC", "CRP", "AGE"]:
    r = {"Feature": LABELS[v]}
    for t in ["COMPLICATED", "ABSCESS"]:
        auc = roc_auc_score(df[t], df[v]); lo, hi = delong_ci(df[t], df[v])
        z = (df[v] - df[v].mean()) / df[v].std()
        f = logit_fit(z.values[:, None], df[t], ["z"])["z"]
        r[f"{t}_AUROC"] = auc; r[f"{t}_lo"] = lo; r[f"{t}_hi"] = hi
        r[f"{t}_ORSD"] = f["OR"]; r[f"{t}_p"] = f["p"]
    uni.append(r)
uni = pd.DataFrame(uni)
uni.to_csv(os.path.join(TAB_DIR, "table4_univariate.csv"), index=False)
print(uni.round(3).to_string())

# ------------------------------------------------------------ Figure 1
fig, ax = plt.subplots(figsize=(7, 4.6))
pct = ct.div(ct.sum(1), axis=0) * 100
cols = ["#cfe1f2", "#7fb2d9", "#e58b5b", "#b2302b"]
left = np.zeros(3)
names = ["Group 1 (score 5-6)", "Group 2 (score 7-8)", "Group 3 (score 9-10)"]
for i, gr in enumerate([1, 2, 3, 4]):
    vals = pct[gr].values if gr in pct else np.zeros(3)
    ax.barh(names, vals, left=left, color=cols[i], edgecolor="white",
            label=["Grade 1 (mild)", "Grade 2 (phlegmonous)", "Grade 3 (gangrenous)", "Grade 4 (perforated)"][i])
    for j, v in enumerate(vals):
        if v >= 7:
            ax.text(left[j] + v / 2, j, f"{v:.0f}%\n(n={ct.loc[j+1, gr]})", ha="center", va="center", fontsize=8)
    left += vals
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Patients within Alvarado risk group (%)")
ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
ax.set_title(f"Histopathological grade by Alvarado risk group (n=420)\n"
             f"chi-square={chi2:.1f}, p<0.001; Cochran-Armitage trend p<0.001", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure1.png"), dpi=300); plt.close()

# ------------------------------------------------------------ Figure 2 (correlation)
vars_ = ["AGE", "MALE", "WBC", "CRP", "ALVARADO_GROUP", "COMPLICATED", "ABSCESS"]
lab = [LABELS.get(v, v) for v in vars_[:-2]] + ["Complicated", "Abscess"]
cm = df[vars_].corr(method="spearman")
plt.figure(figsize=(6.4, 5.4))
mask = np.triu(np.ones_like(cm, bool), 1)
sns.heatmap(cm, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            xticklabels=lab, yticklabels=lab, annot_kws={"size": 8.5}, cbar_kws={"label": "Spearman rho"})
plt.title("Spearman correlations among predictors and outcomes (n=420)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure2.png"), dpi=300); plt.close()
out["corr"] = {"wbc_alv": float(cm.loc["WBC", "ALVARADO_GROUP"]),
               "crp_alv": float(cm.loc["CRP", "ALVARADO_GROUP"]), "wbc_crp": float(cm.loc["WBC", "CRP"])}

with open(os.path.join(TAB_DIR, "r01_results.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out["alvarado_histo"], indent=1, default=float))
print(out["corr"], out["sample"])
