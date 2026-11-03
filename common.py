"""
common.py - Shared data loading, cleaning and feature definitions (revision v2).

Study: "The relationship between the Alvarado risk group and final
histopathological findings in acute appendicitis" (BMC Surgery, revision).

Single source of truth for every analysis script in this folder.

Data note
---------
Laboratory predictors (leukocyte count, absolute neutrophil count, CRP) were
verified against the hospital laboratory information system. No lymphocyte
count is available, so lymphocyte-based indices (e.g. NLR) are not computed.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Data file is NOT distributed with the code (patient confidentiality).
# Set APPENDICITIS_DATA or place 'veriseti.xlsx' two levels above this folder.
DATA_PATH = os.environ.get(
    "APPENDICITIS_DATA",
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "veriseti.xlsx"))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
OOF_DIR = os.path.join(OUT_DIR, "oof")
for d in (OUT_DIR, FIG_DIR, TAB_DIR, OOF_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
SEEDS = (42, 1, 7, 2024, 99)          # five repetitions of nested CV
N_OUTER, N_INNER, N_ITER = 5, 3, 20   # nested CV settings

RAW_COLUMNS = ["PATIENT_ID", "AGE", "SEX", "WBC", "NEUT", "ALVARADO_GROUP",
               "SURGERY", "LOS", "ABSCESS_RAW", "CRP", "HISTO"]

# ---------------------------------------------------------------- feature sets
# PRIMARY: verified routine preoperative variables
CORE = ["AGE", "MALE", "WBC", "NEUT", "CRP", "ALVARADO_GROUP"]
# Sensitivity 1 (R1-8): laboratory/demographic variables WITHOUT Alvarado
CORE_NO_ALV = ["AGE", "MALE", "WBC", "NEUT", "CRP"]
# Sensitivity 2: primary set without the neutrophil count
CORE_NO_NEUT = ["AGE", "MALE", "WBC", "CRP", "ALVARADO_GROUP"]
FEATURE_SETS = {"core": CORE, "core_no_alvarado": CORE_NO_ALV, "core_no_neutrophil": CORE_NO_NEUT}

OUTCOMES = {"COMPLICATED": "Complicated appendicitis (grade 3-4)",
            "ABSCESS": "Intraoperative abscess",
            "COMPOSITE": "Grade 3-4 or intraoperative abscess"}

LABELS = {"AGE": "Age", "MALE": "Male sex", "WBC": "Leukocyte", "NEUT": "Neutrophil", "CRP": "CRP",
          "ALVARADO_GROUP": "Alvarado group", "COMPLICATED": "Complicated", "ABSCESS": "Abscess"}


def load_clean(verbose=False):
    df = pd.read_excel(DATA_PATH)
    df.columns = RAW_COLUMNS
    log = {}
    assert (df.NEUT <= df.WBC).all()

    ab = df["ABSCESS_RAW"].astype(str).str.upper().str.strip()
    log["abscess_missing"] = int(df["ABSCESS_RAW"].isna().sum())
    # one missing abscess field: operative note did not report an abscess -> absent
    df["ABSCESS"] = (ab == "APSE").astype(int)
    df["MALE"] = (df["SEX"].astype(str).str.upper().str.strip() == "E").astype(int)
    df["LAPAROSCOPIC"] = df["SURGERY"].astype(str).str.upper().str.contains("LAPAROSKOP").astype(int)

    df["COMPLICATED"] = (df["HISTO"] >= 3).astype(int)
    df["COMPOSITE"] = ((df["HISTO"] >= 3) | (df["ABSCESS"] == 1)).astype(int)
    df["LOS"] = df["LOS"].astype(float)

    if verbose:
        print(log)
    df.attrs["cleaning_log"] = log
    return df


if __name__ == "__main__":
    d = load_clean(verbose=True)
    print(d.shape)
    for t in OUTCOMES:
        print(t, int(d[t].sum()), f"{d[t].mean()*100:.1f}%")
