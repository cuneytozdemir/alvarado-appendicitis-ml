"""r07_additional_file.py - Build 'Additional file 1' (supplementary tables + learning curve) as .docx."""
import os, json, sys
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from common import TAB_DIR, FIG_DIR, OUT_DIR

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "Additional_file_1.docx")
T = pd.read_csv(os.path.join(TAB_DIR, "all_models_performance.csv"))
lc = json.load(open(os.path.join(TAB_DIR, "learning_curve_samplesize.json")))

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Cm(1.8))


def shade(cell, hexcolor="D9E2F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def table(df, widths=None, fs=8):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""
        r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(fs); shade(cell)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if pd.isna(v) else str(v)); r.font.size = Pt(fs)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def caption(text):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = True; r.font.size = Pt(10)


def note(text):
    p = doc.add_paragraph(); r = p.add_run(text); r.italic = True; r.font.size = Pt(8)


def f3(x):
    return "" if pd.isna(x) else f"{x:.3f}"


def fp(x):
    if pd.isna(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


h = doc.add_paragraph(); r = h.add_run("Additional file 1"); r.bold = True; r.font.size = Pt(14)
doc.add_paragraph("The Relationship Between the Alvarado Risk Group and Final Histopathological Findings in Acute "
                  "Appendicitis, with Internally Validated Machine-Learning Models for Complicated Appendicitis "
                  "and Intraoperative Abscess — supplementary tables and figure.")

FSN = {"core": "Primary: age, sex, leukocyte, CRP, Alvarado group",
       "core_no_alvarado": "Without Alvarado: age, sex, leukocyte, CRP",
       "alvarado": "Alvarado risk group alone"}
OUTN = {"COMPLICATED": "Complicated appendicitis (histopathological grade 3–4)",
        "ABSCESS": "Intraoperative abscess",
        "COMPOSITE": "Composite outcome (grade 3–4 or intraoperative abscess)"}

# ---------------- Table S1: all algorithms, primary feature set
for k, outcome in enumerate(["COMPLICATED", "ABSCESS"]):
    caption(f"Table S{k+1}. Performance of all evaluated algorithms – {OUTN[outcome]} "
            f"(primary feature set: age, sex, leukocyte count, CRP, Alvarado risk group).")
    sub = T[(T.outcome == outcome) & T.feature_set.isin(["core", "alvarado"])].copy()
    sub["rank"] = sub.model.map(lambda m: 0 if m == "Random forest" else (2 if m.startswith("Alvarado") else 1))
    sub = sub.sort_values(["rank", "AUROC"], ascending=[True, False])
    rows = []
    for _, r in sub.iterrows():
        name = r.model + (" (main model)" if r.model == "Random forest" else "") + \
            (" (reference)" if r.model.startswith("Alvarado") else "")
        rows.append({"Model": name,
                     "AUROC (95% CI)†": f"{r.AUROC:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f})",
                     "AUROC per repetition, mean ± SD (range)": f"{r.AUROC_rep_mean:.3f} ± {r.AUROC_rep_sd:.3f} ({r.AUROC_rep_min:.3f}–{r.AUROC_rep_max:.3f})",
                     "AUPRC": f3(r.AUPRC),
                     "Sensitivity‡": f3(r.get("Sensitivity")), "Specificity‡": f3(r.get("Specificity")),
                     "Brier": f3(r.get("Brier")),
                     "ΔAUROC vs Alvarado (95% CI)": "—" if pd.isna(r.get("dAUROC")) else f"{r.dAUROC:+.3f} ({r.dAUROC_lo:+.3f} to {r.dAUROC_hi:+.3f})",
                     "DeLong p (raw)": fp(r.get("p_DeLong_vs_Alvarado")),
                     "Holm p (family)": "—" if pd.isna(r.p_Holm) else f"{fp(r.p_Holm)} ({'A' if r.family.startswith('A') else 'B'})",
                     "Repetitions with p<0.05": "—" if pd.isna(r.get("n_rep_sig")) else f"{int(r.n_rep_sig)}/5"})
    table(pd.DataFrame(rows), widths=[4.2, 3.2, 4.4, 1.4, 1.6, 1.6, 1.3, 3.6, 1.6, 2.0, 1.6], fs=7.5)
    note("† Computed from one averaged out-of-fold prediction per patient (mean of the five repetitions of 5-fold nested "
         "cross-validation); 95% CI from 2000 patient-level bootstrap resamples. ‡ At a Youden threshold selected on "
         "inner-cross-validation predictions of the outer-training data only and applied to the held-out fold; mean over "
         "five repetitions (not defined for the averaging ensemble). Family A = primary comparison (main random-forest model "
         "vs Alvarado for the two outcomes; Holm, m=2); family B = exploratory comparisons (ten other models vs Alvarado per "
         "outcome; Holm, m=10). The Alvarado reference uses the ordinal Alvarado group as the score (no fitted parameters); "
         "its Brier score is reported in the calibration analysis.")
    doc.add_paragraph()

# ---------------- Table S3: sensitivity analyses
caption("Table S3. Sensitivity analyses: predictor set without the Alvarado group, and composite outcome (random forest and logistic regression).")
sub = T[T.feature_set.isin(["core", "core_no_alvarado"]) &
        T.model.isin(["Random forest", "Logistic regression"])].copy()
sub["o"] = sub.outcome.map({"COMPLICATED": 0, "ABSCESS": 1, "COMPOSITE": 2})
sub["f"] = sub.feature_set.map({"core": 0, "core_no_alvarado": 1})
sub = sub.sort_values(["o", "f", "model"], ascending=[True, True, False])
rows = []
for _, r in sub.iterrows():
    rows.append({"Outcome": OUTN[r.outcome].split(" (")[0], "Predictor set": FSN[r.feature_set], "Model": r.model,
                 "AUROC (95% CI)": f"{r.AUROC:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f})",
                 "Per-repetition mean ± SD": f"{r.AUROC_rep_mean:.3f} ± {r.AUROC_rep_sd:.3f}",
                 "ΔAUROC vs Alvarado (95% CI)": f"{r.dAUROC:+.3f} ({r.dAUROC_lo:+.3f} to {r.dAUROC_hi:+.3f})",
                 "DeLong p vs Alvarado (unadjusted)": fp(r.p_DeLong_vs_Alvarado)})
alv = T[T.model == "Alvarado risk group"]
for _, r in alv.iterrows():
    rows.append({"Outcome": OUTN[r.outcome].split(" (")[0], "Predictor set": FSN["alvarado"], "Model": "—",
                 "AUROC (95% CI)": f"{r.AUROC:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f})",
                 "Per-repetition mean ± SD": "—", "ΔAUROC vs Alvarado (95% CI)": "—",
                 "DeLong p vs Alvarado (unadjusted)": "—"})
table(pd.DataFrame(rows), widths=[3.4, 6.0, 3.0, 3.4, 3.0, 4.0, 3.0], fs=7.5)
note("The primary predictor set and the set without the Alvarado group are shown for both algorithms; the composite outcome was analysed with the primary set only.")
pw = pd.read_csv(os.path.join(TAB_DIR, "pairwise_sensitivity.csv"))
doc.add_paragraph()
caption("Table S4. Direct DeLong comparisons between predictor sets (random forest).")
pw["Outcome"] = pw.outcome.map(lambda o: OUTN[o].split(" (")[0])
pw["Comparison"] = pw.comparison.map({
    "RF core vs RF core_no_alvarado": "Primary set vs set without Alvarado group",
    "RF core vs LR core": "Random forest vs logistic regression (primary set)"})
table(pd.DataFrame({"Outcome": pw.Outcome, "Comparison": pw.Comparison, "AUROC A": pw.auc_a.map(f3),
                    "AUROC B": pw.auc_b.map(f3), "DeLong p": pw.p.map(fp)}), widths=[4, 9, 2.5, 2.5, 2.5], fs=8)

# ---------------- Table S5: hyper-parameter spaces
doc.add_paragraph()
caption("Table S5. Hyper-parameter search spaces (inner loop: stratified 3-fold cross-validation, randomized search with 20 iterations, AUROC as criterion).")
hp = [
    ("Logistic regression", "Standardization; class_weight = balanced", "C ~ log-uniform(0.01, 100)"),
    ("ElasticNet logistic regression", "Standardization; saga solver; class_weight = balanced", "C ~ log-uniform(0.01, 10); l1_ratio ~ uniform(0, 1)"),
    ("SVM (RBF kernel)", "Standardization; Platt probabilities; class_weight = balanced", "C ~ log-uniform(0.1, 100); gamma ~ log-uniform(0.001, 1)"),
    ("k-nearest neighbours", "Standardization", "k ~ integer(3, 24); weights ∈ {uniform, distance}"),
    ("Random forest (main model)", "class_weight = balanced", "n_estimators ~ integer(100, 399); max_depth ~ integer(3, 11); min_samples_leaf ~ integer(1, 11)"),
    ("Extra trees", "class_weight = balanced", "n_estimators ~ integer(100, 399); max_depth ~ integer(3, 11); min_samples_leaf ~ integer(1, 11)"),
    ("XGBoost", "scale_pos_weight = negatives/positives of the training fold", "n_estimators ~ integer(100, 399); max_depth ~ integer(2, 6); learning_rate ~ log-uniform(0.01, 0.3); subsample ~ uniform(0.6, 1.0); colsample_bytree ~ uniform(0.6, 1.0)"),
    ("LightGBM", "class_weight = balanced", "n_estimators ~ integer(100, 399); max_depth ~ integer(2, 7); learning_rate ~ log-uniform(0.01, 0.3); num_leaves ~ integer(8, 47)"),
    ("CatBoost", "auto_class_weights = Balanced", "depth ∈ {2,…,6}; learning_rate ∈ 25 log-spaced values in [0.01, 0.3]; iterations ∈ {100, 125, …, 400}"),
    ("Multilayer perceptron", "Standardization; early stopping", "hidden layers ∈ {(32), (64), (64,32), (32,16)}; alpha ~ log-uniform(0.0001, 0.1)"),
    ("Ensemble", "Unweighted mean of the out-of-fold probabilities of random forest, XGBoost, LightGBM and CatBoost", "—"),
    ("Isotonic recalibration (main model)", "CalibratedClassifierCV (isotonic, inner 3-fold) fitted on outer-training data only", "—"),
]
table(pd.DataFrame(hp, columns=["Algorithm", "Fixed settings / preprocessing", "Search space"]), widths=[5, 8, 13], fs=8)

# ---------------- Table S6: sample size
doc.add_paragraph()
caption("Table S6. Events per variable and minimum sample size according to Riley et al. (BMJ 2020).")
rows = []
for k, v in lc["sample_size"].items():
    o, s = k.split("|")
    if "core" not in s:
        continue
    rows.append({"Outcome": OUTN[o].split(" (")[0], "Predictor set (parameters)": s, "Events": v["events"],
                 "EPV": f"{v['EPV']:.1f}", "Required n (criterion 1: shrinkage ≥0.9)": f"{v['n_criterion1']:.0f}",
                 "Criterion 2": f"{v['n_criterion2']:.0f}", "Criterion 3": f"{v['n_criterion3']:.0f}",
                 "Minimum n required": f"{v['n_required']:.0f}", "Available n": "420"})
table(pd.DataFrame(rows), fs=8)
note("Assumes an anticipated Cox–Snell R² equal to 15% of its maximum (conservative default recommended by Riley et al.); "
     "the Alvarado risk group is counted as one ordinal parameter.")

# ---------------- Figure S1
doc.add_page_break()
caption("Figure S1. Learning curves of the main random-forest model.")
doc.add_picture(os.path.join(FIG_DIR, "figureS1_learning_curve.png"), width=Cm(22))
note("The model (primary feature set, median of the hyper-parameters selected in the nested cross-validation) was trained on "
     "20–100% of each training split of a stratified 5-fold cross-validation repeated 10 times (50 splits); AUROC was always "
     "evaluated on the complete held-out fold. Points: mean; error bars: SD.")
doc.save(OUT)
print("saved", OUT)
