"""
common.py - Shared data loading, cleaning and feature definitions (revision v2).

Study: "The relationship between the Alvarado risk group and final
histopathological findings in acute appendicitis" (BMC Surgery, revision).

Single source of truth for every analysis script in this folder.

Data note (reviewer comments R1-3, R2, R3-1, R4-3)
--------------------------------------------------
The column labelled 'LENFOSIT' (lymphocyte) in the source spreadsheet was
mislabelled. On verification against the hospital laboratory information
system for all patients it was confirmed to be the ABSOLUTE NEUTROPHIL COUNT
(cells/uL). The laboratory records contain no lymphocyte count, so no
lymphocyte-based index (e.g. NLR) can be or is computed.

During verification, four transcription errors (a missing trailing zero) were
identified and are corrected below from the laboratory records (CORRECTIONS).
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

# Transcription errors corrected from the laboratory information system
# (patient id -> {column: verified value}); a trailing zero was missing.
CORRECTIONS = {26: {"NEUT": 12860}, 33: {"NEUT": 12910}, 48: {"NEUT": 15950}, 330: {"WBC": 17000}}

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
    for pid, fix in CORRECTIONS.items():
        for col, val in fix.items():
            df.loc[df.PATIENT_ID == pid, col] = val
    log["corrections"] = len(CORRECTIONS)
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
