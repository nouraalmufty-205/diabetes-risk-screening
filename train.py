"""Train, tune, evaluate and save the diabetes screening models.

    python train.py --data data/diabetes_binary_health_indicators_BRFSS2015.csv

Every number in outputs/ comes from this code. Nothing is hard-coded.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from utils import ALL_FEATURES, TARGET, find_threshold, load_data, make_pipeline, reference_values

RS = 42
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": loguniform(1e-2, 1e1)},
    ),
    "Random Forest": (
        RandomForestClassifier(n_jobs=-1, random_state=RS),
        {"n_estimators": [100, 200], "max_depth": [8, 12, 16], "min_samples_leaf": [5, 10, 20]},
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=RS),
        {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [100, 200, 300],
         "max_leaf_nodes": [15, 31], "min_samples_leaf": [20, 50, 100]},
    ),
}


def scores(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
        "recall": recall_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred), "threshold": thr,
    }


def tune(name, X_train, y_train, n_tune, n_iter):
    """Randomised search with 5-fold CV on (a sample of) the TRAINING data only."""
    est, grid = CANDIDATES[name]
    if len(X_train) > n_tune:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=n_tune,
                                                  stratify=y_train, random_state=RS)
    search = RandomizedSearchCV(
        make_pipeline(ALL_FEATURES, clone(est)), {f"model__{k}": v for k, v in grid.items()},
        n_iter=n_iter, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
        scoring="roc_auc", n_jobs=-1, random_state=RS)
    return search.fit(X_train, y_train)


def fit_with_threshold(features, estimator, X_train, y_train, target_recall):
    """Fit on 80% of train, pick the threshold on the other 20%. Returns (val_model, thr, X_val, y_val)."""
    X_a, X_v, y_a, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RS)
    m = make_pipeline(features, clone(estimator)).fit(X_a[features], y_a)
    return m, find_threshold(y_v, m.predict_proba(X_v[features])[:, 1], target_recall), X_v, y_v


def main(a):
    out, models = Path(a.out), Path(a.models)
    out.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    df = load_data(a.data)
    n_raw = len(df)
    df = df.drop_duplicates()  # drop exact duplicate rows BEFORE splitting
    X, y = df[ALL_FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    print(f"rows: {n_raw} -> {len(df)} after removing duplicates | train {len(X_train)} test {len(X_test)}")

    results, fitted = {}, {}
    base = make_pipeline(ALL_FEATURES, DummyClassifier(strategy="prior")).fit(X_train, y_train)
    results["Baseline (majority class)"] = {**scores(y_test, base.predict_proba(X_test)[:, 1], 0.5), "cv_roc_auc": 0.5}

    for name in CANDIDATES:
        print(f"tuning {name} ...")
        s = tune(name, X_train, y_train, a.n_tune, a.n_iter)
        est = clone(CANDIDATES[name][0]).set_params(**{k.replace("model__", ""): v for k, v in s.best_params_.items()})
        _, thr, _, _ = fit_with_threshold(ALL_FEATURES, est, X_train, y_train, a.target_recall)
        final = make_pipeline(ALL_FEATURES, est).fit(X_train, y_train)
        proba = final.predict_proba(X_test)[:, 1]
        results[name] = {**scores(y_test, proba, thr), "cv_roc_auc": s.best_score_,
                         "recall_at_0.5": recall_score(y_test, proba >= 0.5), "best_params": s.best_params_}
        fitted[name] = (est, thr, final)

    best = max(fitted, key=lambda n: results[n]["cv_roc_auc"])
    est, thr, final = fitted[best]
    print(f"best model by CV ROC-AUC: {best}")

    # --- which questions matter? permutation importance on a validation split
    m_val, thr_v, X_v, y_v = fit_with_threshold(ALL_FEATURES, est, X_train, y_train, a.target_recall)
    Xs = X_v.sample(min(len(X_v), 20000), random_state=RS)
    imp = permutation_importance(m_val, Xs, y_v.loc[Xs.index], scoring="roc_auc", n_repeats=3,
                                 random_state=RS, n_jobs=-1)
    ranking = pd.Series(imp.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
    ranking.to_csv(out / "feature_importance.csv", header=["roc_auc_drop"])

    # --- stretch goal: how few questions keep the recall?
    k_rows, short = [], None
    for k in sorted({3, 5, 7, 10, 15, len(ALL_FEATURES), a.k}):
        feats = list(ranking.index[:k])
        m, t, _, _ = fit_with_threshold(feats, est, X_train, y_train, a.target_recall)
        k_rows.append({"n_questions": k, **scores(y_test, m.predict_proba(X_test[feats])[:, 1], t)})
    pd.DataFrame(k_rows).to_csv(out / "questions_vs_recall.csv", index=False)

    feats = list(ranking.index[: a.k])
    _, t_short, _, _ = fit_with_threshold(feats, est, X_train, y_train, a.target_recall)
    short_model = make_pipeline(feats, clone(est)).fit(X_train[feats], y_train)
    short_scores = scores(y_test, short_model.predict_proba(X_test[feats])[:, 1], t_short)
    results[f"{best} - short form ({a.k} questions)"] = {**short_scores, "cv_roc_auc": None}

    for fname, pipe, f, t, sc in [("model_full.joblib", final, ALL_FEATURES, thr, results[best]),
                                  ("model_short.joblib", short_model, feats, t_short, short_scores)]:
        joblib.dump({"pipeline": pipe, "features": f, "threshold": t,
                     "reference": reference_values(X_train[f]), "model_name": best, "test_scores": sc},
                    models / fname)

    table = pd.DataFrame(results).T.drop(columns=["best_params"], errors="ignore")
    table.to_csv(out / "results.csv")
    (out / "metrics.json").write_text(json.dumps({
        "rows_raw": n_raw, "rows_after_dedup": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
        "positive_rate": float(y.mean()), "best_model": best, "target_recall": a.target_recall,
        "short_form_features": feats, "results": results}, indent=2, default=float))
    print(table.round(3).to_string())
    print("short form questions:", feats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/diabetes_binary_health_indicators_BRFSS2015.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--models", default="models")
    p.add_argument("--k", type=int, default=7, help="questions in the short form")
    p.add_argument("--n-tune", type=int, default=40000, help="rows used for hyper-parameter search")
    p.add_argument("--n-iter", type=int, default=8, help="random-search iterations per model")
    p.add_argument("--target-recall", type=float, default=0.80)
    main(p.parse_args())
