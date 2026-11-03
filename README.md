# Alvarado risk group, histopathology and machine-learning prediction of complicated appendicitis / intraoperative abscess

Analysis code accompanying the revised manuscript
*"The Relationship Between the Alvarado Risk Group and Final Histopathological Findings in Acute Appendicitis, with Internally Validated Machine-Learning Models for Complicated Appendicitis and Intraoperative Abscess"* (BMC Surgery, under review).

## Data
Patient-level data cannot be shared publicly (confidentiality). Place the de-identified spreadsheet
(`veriseti.xlsx`, 420 rows, 11 columns) two folders above this directory or point to it with the
environment variable `APPENDICITIS_DATA`. Column order: patient ID, age, sex (E/K), leukocyte count,
an unverifiable count column labelled "lymphocyte" (dropped on loading, not used in any analysis),
Alvarado risk group (1 = score 5-6, 2 = 7-8, 3 = 9-10), surgical approach, length of stay,
intraoperative abscess (APSE/YOK), CRP, histopathological grade (1-4).

## Run
```bash
pip install -r requirements.txt
python run_all.py            # all steps (about 30-60 min on 16 cores)
python run_all.py r03_aggregate   # single step
```

| Script | Content |
|---|---|
| `common.py` | data loading/cleaning, outcome definitions, feature sets |
| `stats_utils.py` | DeLong test, patient-level bootstrap, Holm, calibration slope/intercept, net benefit, logistic MLE |
| `r01_descriptive.py` | Table 2-4, 7, Alvarado x histopathology tests, odds ratios, Figures 1-2 |
| `r02_nested_cv.py` | 5x5 repeated nested CV for 10 algorithms (+ensemble); train-fold-only thresholds and isotonic recalibration |
| `r03_aggregate.py` | averaging of repeated OOF predictions (one per patient), DeLong, bootstrap CI, Holm families, Additional file tables, Figure 4 |
| `r04_calibration_dca.py` | calibration slope/intercept, Brier, decision-curve analysis (Figure 5) |
| `r05_explain.py` | out-of-fold permutation importance (Figure 3), SHAP (Figure 6) |
| `r06_learning_curve_samplesize.py` | learning curve (Additional file 1, Figure S1) and Riley minimum sample size |
| `r07_additional_file.py` | builds Additional file 1 (.docx) |

## Key design choices
* **Predictors** = age, sex, leukocyte count, CRP, Alvarado risk group (sensitivity analysis: without the Alvarado group). The column of unverifiable identity is excluded from all analyses.
* **Repeated predictions**: each patient receives one out-of-fold prediction per repetition; the five are averaged so every patient contributes one prediction to DeLong tests, bootstrap CIs, calibration and DCA.
* **Multiplicity**: Family A (main RF model vs Alvarado for two outcomes, Holm m=2); Family B (ten exploratory models vs Alvarado per outcome, Holm m=10).
* **No leakage**: scaling, class weighting, hyper-parameters, classification thresholds and isotonic recalibration are learned on outer-training folds only.

## License
MIT
