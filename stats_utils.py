"""stats_utils.py - DeLong test, patient-level bootstrap, Holm, calibration metrics, DCA."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p_a, p_b):
    """Two-sided DeLong test for two correlated AUROCs (one prediction per patient)."""
    y = np.asarray(y).astype(int)
    order = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    m = int(y.sum())
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y).astype(int)
    order = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    aucs, cov = _fast_delong(np.asarray(p)[order][None, :], int(y.sum()))
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0] - z * se), float(aucs[0] + z * se)


# ------------------------------------------------------------- bootstrap (patient level)
def bootstrap_ci(y, p, metric=roc_auc_score, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        s = rng.randint(0, len(y), len(y))
        if y[s].min() == y[s].max():
            continue
        vals.append(metric(y[s], p[s]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def bootstrap_diff_ci(y, p_a, p_b, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    d = []
    for _ in range(n):
        s = rng.randint(0, len(y), len(y))
        if y[s].min() == y[s].max():
            continue
        d.append(roc_auc_score(y[s], p_a[s]) - roc_auc_score(y[s], p_b[s]))
    return tuple(np.percentile(d, [2.5, 97.5]))


# ------------------------------------------------------------- Holm
def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = adj
    return out


# ------------------------------------------------------------- calibration
def calibration_slope_intercept(y, p):
    """Slope: logistic recalibration coefficient of logit(p).
    Intercept: calibration-in-the-large (logit(p) as offset)."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]
    # intercept with offset: solve by Newton iterations
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu))
        a += g / h
        if abs(g / h) < 1e-8:
            break
    return float(slope), float(a)


# ------------------------------------------------------------- decision curve
def net_benefit(y, p, t):
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    n = len(y)
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return tp / n - fp / n * (t / (1 - t))


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    t = thr[j]
    if not np.isfinite(t):          # roc_curve's first threshold is +inf
        t = np.max(p) + 1e-9
    return float(t)


# ------------------------------------------------------------- logistic regression (MLE + Wald CI)
def logit_fit(X, y, names=None):
    """Unpenalised logistic regression via Newton-Raphson; returns dict of OR, 95% CI, p."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ b))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = b / se
    p = 2 * stats.norm.sf(np.abs(z))
    names = names or [f"x{i}" for i in range(1, X.shape[1])]
    return {n: dict(OR=float(np.exp(b[i + 1])), lo=float(np.exp(b[i + 1] - 1.96 * se[i + 1])),
                    hi=float(np.exp(b[i + 1] + 1.96 * se[i + 1])), p=float(p[i + 1]))
            for i, n in enumerate(names)}
