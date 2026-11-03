"""
r02_nested_cv.py - Repeated nested cross-validation for all models (revision v2).

Design (addresses R1-6, R2 statistics, R3-4, R4-6)
  * Outer loop : stratified 5-fold -> out-of-fold (OOF) predictions
  * Inner loop : stratified 3-fold RandomizedSearchCV (20 iterations, AUROC)
  * Repeated with 5 random seeds -> each patient receives exactly one OOF
    prediction per repetition (5 per patient in total).
  * Everything data-dependent (scaling, class weights, hyper-parameters,
    isotonic recalibration and the classification threshold) is learned on the
    outer-training data only:
      - threshold  : Youden's J on inner-CV OOF predictions of the training data
      - isotonic   : CalibratedClassifierCV(cv=3) fitted on the training data
  * Alvarado reference: the ordinal Alvarado group itself is used as the score
    for discrimination (no fitting needed); a fold-wise logistic regression on
    the training data converts it to probabilities for calibration / DCA.

Output: outputs/oof/nested_results.pkl
"""
import os, pickle, time, warnings
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from common import (load_clean, FEATURE_SETS, SEEDS, N_OUTER, N_INNER, N_ITER, OOF_DIR)
from stats_utils import youden_threshold

warnings.filterwarnings("ignore")


def make_models(seed):
    sc = [("scaler", StandardScaler())]
    return {
        "Logistic regression": (Pipeline(sc + [("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
                                {"clf__C": stats.loguniform(1e-2, 1e2)}),
        "ElasticNet": (Pipeline(sc + [("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, class_weight="balanced", random_state=seed))]),
                       {"clf__C": stats.loguniform(1e-2, 1e1), "clf__l1_ratio": stats.uniform(0, 1)}),
        "SVM (RBF)": (Pipeline(sc + [("clf", SVC(probability=True, class_weight="balanced", random_state=seed))]),
                      {"clf__C": stats.loguniform(1e-1, 1e2), "clf__gamma": stats.loguniform(1e-3, 1e0)}),
        "k-nearest neighbours": (Pipeline(sc + [("clf", KNeighborsClassifier())]),
                                 {"clf__n_neighbors": stats.randint(3, 25), "clf__weights": ["uniform", "distance"]}),
        "Random forest": (Pipeline([("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1))]),
                          {"clf__n_estimators": stats.randint(100, 400), "clf__max_depth": stats.randint(3, 12),
                           "clf__min_samples_leaf": stats.randint(1, 12)}),
        "Extra trees": (Pipeline([("clf", ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=1))]),
                        {"clf__n_estimators": stats.randint(100, 400), "clf__max_depth": stats.randint(3, 12),
                         "clf__min_samples_leaf": stats.randint(1, 12)}),
        "XGBoost": (Pipeline([("clf", XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0))]),
                    {"clf__n_estimators": stats.randint(100, 400), "clf__max_depth": stats.randint(2, 7),
                     "clf__learning_rate": stats.loguniform(1e-2, 3e-1), "clf__subsample": stats.uniform(.6, .4),
                     "clf__colsample_bytree": stats.uniform(.6, .4)}),
        "LightGBM": (Pipeline([("clf", LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1))]),
                     {"clf__n_estimators": stats.randint(100, 400), "clf__max_depth": stats.randint(2, 8),
                      "clf__learning_rate": stats.loguniform(1e-2, 3e-1), "clf__num_leaves": stats.randint(8, 48)}),
        "CatBoost": (Pipeline([("clf", CatBoostClassifier(verbose=0, random_state=seed, auto_class_weights="Balanced", thread_count=1, allow_writing_files=False))]),
                     {"clf__depth": [2, 3, 4, 5, 6],  # python-native values: CatBoost cannot be sklearn-cloned with numpy scalars
                      "clf__learning_rate": [float(round(v, 4)) for v in np.geomspace(1e-2, 3e-1, 25)],
                      "clf__iterations": list(range(100, 401, 25))}),
        "Multilayer perceptron": (Pipeline(sc + [("clf", MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed))]),
                                  {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (32, 16)],
                                   "clf__alpha": stats.loguniform(1e-4, 1e-1)}),
    }


def run_task(df, outcome, fs_name, model_name, seed, fold, calibrate):
    X = df[FEATURE_SETS[fs_name]]
    y = df[outcome].values
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    tr, te = list(outer.split(X, y))[fold]
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    est, dist = make_models(seed)[model_name]
    if model_name == "XGBoost":
        est.set_params(clf__scale_pos_weight=(len(tr) - y[tr].sum()) / y[tr].sum())
    search = RandomizedSearchCV(est, dist, n_iter=N_ITER, scoring="roc_auc", cv=inner,
                                random_state=seed, n_jobs=1, refit=True)
    search.fit(X.iloc[tr], y[tr])
    best = search.best_estimator_

    def fresh():  # re-instantiate (CatBoost cannot be sklearn-cloned)
        e, _ = make_models(seed)[model_name]
        e.set_params(**search.best_params_)
        if model_name == "XGBoost":
            e.set_params(clf__scale_pos_weight=(len(tr) - y[tr].sum()) / y[tr].sum())
        return e
    p_te = best.predict_proba(X.iloc[te])[:, 1]
    # threshold learned on training data only (inner-CV OOF predictions)
    p_tr_oof = cross_val_predict(fresh(), X.iloc[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
    thr = youden_threshold(y[tr], p_tr_oof)
    out = dict(outcome=outcome, fs=fs_name, model=model_name, seed=seed, fold=fold,
               te=te, p=p_te, thr=thr, best_params={k: (v.item() if hasattr(v, "item") else v)
                                                    for k, v in search.best_params_.items()})
    if calibrate:
        cal = CalibratedClassifierCV(fresh(), method="isotonic", cv=inner)
        cal.fit(X.iloc[tr], y[tr])
        out["p_cal"] = cal.predict_proba(X.iloc[te])[:, 1]
        p_tr_cal = cross_val_predict(CalibratedClassifierCV(fresh(), method="isotonic", cv=N_INNER),
                                     X.iloc[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
        out["thr_cal"] = youden_threshold(y[tr], p_tr_cal)
    return out


def alvarado_task(df, outcome, seed, fold):
    y = df[outcome].values
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    tr, te = list(outer.split(df, y))[fold]
    a = df["ALVARADO_GROUP"].values.astype(float)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(a[tr, None], y[tr])
    p_tr = lr.predict_proba(a[tr, None])[:, 1]
    return dict(outcome=outcome, fs="alvarado", model="Alvarado risk group", seed=seed, fold=fold,
                te=te, p=a[te], p_cal=lr.predict_proba(a[te, None])[:, 1],
                thr=youden_threshold(y[tr], a[tr]), thr_cal=youden_threshold(y[tr], p_tr))


if __name__ == "__main__":
    df = load_clean()
    all_models = list(make_models(0).keys())
    jobs = []
    for seed in SEEDS:
        for fold in range(N_OUTER):
            for outcome in ["COMPLICATED", "ABSCESS", "COMPOSITE"]:
                jobs.append(("alv", outcome, seed, fold))
                # primary feature set: all algorithms for the two main outcomes
                models = all_models if outcome != "COMPOSITE" else ["Random forest", "Logistic regression"]
                for m in models:
                    jobs.append(("ml", outcome, "core", m, seed, fold, m == "Random forest"))
                if outcome == "COMPOSITE":
                    continue
                for fs in ["core_no_alvarado"]:
                    for m in ["Random forest", "Logistic regression"]:
                        jobs.append(("ml", outcome, fs, m, seed, fold, False))
    print(f"{len(jobs)} tasks")
    t0 = time.time()

    def dispatch(j):
        if j[0] == "alv":
            return alvarado_task(df, *j[1:])
        return run_task(df, *j[1:])

    res = Parallel(n_jobs=15, verbose=5)(delayed(dispatch)(j) for j in jobs)
    with open(os.path.join(OOF_DIR, "nested_results.pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"done in {time.time()-t0:.0f}s")
