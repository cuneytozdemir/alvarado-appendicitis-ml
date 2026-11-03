"""
r04_calibration_dca.py - Calibration and decision-curve analysis of the MAIN model
(random forest, core feature set) vs the Alvarado risk group (R2, R3-4, R4-6).

Isotonic recalibration was fitted inside each outer-training fold only
(CalibratedClassifierCV with an inner 3-fold split); the outer test fold never
contributes to recalibration. Calibration metrics are computed on the averaged
out-of-fold predictions (one per patient).
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from common import load_clean, OOF_DIR, TAB_DIR, FIG_DIR
from stats_utils import calibration_slope_intercept, net_benefit

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
df = load_clean()
avg = pickle.load(open(os.path.join(OOF_DIR, "averaged_predictions.pkl"), "rb"))["avg"]
out = {}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
ts = np.linspace(0.02, 0.60, 59)
for j, (outcome, title) in enumerate([("COMPLICATED", "Complicated appendicitis"), ("ABSCESS", "Intraoperative abscess")]):
    y = df[outcome].values
    raw = avg[f"{outcome}|core|Random forest"]
    cal = avg[f"CAL|{outcome}|core|Random forest"]
    alv = avg[f"CAL|{outcome}|alvarado|Alvarado risk group"]   # fold-wise logistic probability
    r = {}
    for lab, p in [("rf_raw", raw), ("rf_isotonic", cal), ("alvarado", alv)]:
        s, a = calibration_slope_intercept(y, p)
        r[lab] = dict(slope=s, intercept=a, brier=brier_score_loss(y, p))
    # net benefit table + bootstrap CI for the difference (model - Alvarado)
    rng = np.random.RandomState(42)
    nbt = {}
    for t in [0.10, 0.20, 0.30, 0.40]:
        d0 = net_benefit(y, cal, t) - net_benefit(y, alv, t)
        boots = []
        for _ in range(2000):
            s_ = rng.randint(0, len(y), len(y))
            boots.append(net_benefit(y[s_], cal[s_], t) - net_benefit(y[s_], alv[s_], t))
        nbt[f"{t:.2f}"] = dict(nb_model=net_benefit(y, cal, t), nb_alvarado=net_benefit(y, alv, t),
                               nb_treat_all=y.mean() - (1 - y.mean()) * t / (1 - t),
                               diff=d0, diff_lo=float(np.percentile(boots, 2.5)), diff_hi=float(np.percentile(boots, 97.5)))
    r["net_benefit"] = nbt
    nb_m = np.array([net_benefit(y, cal, t) for t in ts])
    nb_a = np.array([net_benefit(y, alv, t) for t in ts])
    nb_all = y.mean() - (1 - y.mean()) * ts / (1 - ts)
    sel = (ts >= 0.10) & (ts <= 0.40)
    r["frac_thr_10_40_model_gt_alvarado"] = float(np.mean(nb_m[sel] > nb_a[sel]))
    r["max_abs_nb_diff_10_40"] = float(np.max(np.abs(nb_m[sel] - nb_a[sel])))
    out[outcome] = r

    ax = axes[0, j]
    for p, lab, c, mk in [(raw, f"RF, uncalibrated (slope {r['rf_raw']['slope']:.2f}; Brier {r['rf_raw']['brier']:.3f})", "0.6", "o"),
                          (cal, f"RF, isotonic (slope {r['rf_isotonic']['slope']:.2f}; Brier {r['rf_isotonic']['brier']:.3f})", "#1f4e8c", "s"),
                          (alv, f"Alvarado group (slope {r['alvarado']['slope']:.2f}; Brier {r['alvarado']['brier']:.3f})", "#d9822b", "^")]:
        fr, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mp, fr, marker=mk, color=c, lw=1.8, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title(f"{'AB'[j]}  Calibration - {title}", loc="left", fontsize=10)
    ax.legend(fontsize=7.3, loc="upper left")

    ax = axes[1, j]
    ax.plot(ts, nb_m, color="#1f4e8c", lw=2.4, label="Random forest (isotonic), main model")
    ax.plot(ts, nb_a, color="#d9822b", lw=2, label="Alvarado risk group")
    ax.plot(ts, nb_all, color="0.5", ls="--", lw=1.2, label="Treat all")
    ax.axhline(0, color="k", lw=0.8, label="Treat none")
    ax.set_ylim(-0.05, max(nb_m.max(), y.mean()) + 0.04)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"{'CD'[j]}  Decision curve - {title}", loc="left", fontsize=10)
    ax.legend(fontsize=7.3)
fig.suptitle("Main model (random forest; age, sex, leukocyte count, neutrophil count, CRP, Alvarado risk group). Out-of-fold predictions "
             "from repeated nested cross-validation;\nisotonic recalibration fitted within outer-training folds only.",
             fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "figure5.png"), dpi=300); plt.close()
json.dump(out, open(os.path.join(TAB_DIR, "calibration_dca.json"), "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
