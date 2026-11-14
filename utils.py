"""Shared helpers for the Diabetes Risk Screening project.

Used by train.py, the notebook and the Streamlit app. The app must be able to
import this module so the saved pipeline can be unpickled.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TARGET = "Diabetes_binary"
BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]
CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]
ORDINAL = ["GenHlth", "Age", "Education", "Income"]
ALL_FEATURES = BINARY + CONTINUOUS + ORDINAL


# ---------------------------------------------------------------- data
def load_data(path=None):
    """Load the CDC Diabetes Health Indicators data.

    Uses a local CSV if `path` exists (e.g. the Kaggle copy
    diabetes_binary_health_indicators_BRFSS2015.csv), otherwise downloads it
    from the UCI repository (dataset id 891) with the `ucimlrepo` package.
    """
    if path and Path(path).exists():
        df = pd.read_csv(path)
    else:
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

        data = fetch_ucirepo(id=891)
        df = pd.concat([data.data.features, data.data.targets], axis=1)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[ALL_FEATURES + [TARGET]].astype(float)


def reference_values(X):
    """Typical respondent: median for numeric/ordinal, most common value for yes/no."""
    ref = {}
    for c in X.columns:
        ref[c] = float(X[c].mode().iloc[0]) if c in BINARY else float(X[c].median())
    return ref


# ------------------------------------------------------------ pipeline
def clip_bmi(X):
    """Outlier handling: keep BMI inside a physiologically plausible range."""
    return np.clip(X, 12, 60)


def add_features(X):
    """Feature engineering: total number of 'not good' physical + mental days."""
    X = X.copy()
    if {"MentHlth", "PhysHlth"} <= set(X.columns):
        X["PoorHealthDays"] = X["MentHlth"] + X["PhysHlth"]
    return X


def make_pipeline(features, estimator):
    """Full preprocessing + model pipeline restricted to `features`."""
    f = list(features)
    days = [c for c in ["MentHlth", "PhysHlth"] if c in f]
    if len(days) == 2:
        days.append("PoorHealthDays")

    def num(extra=()):
        return Pipeline([("imp", SimpleImputer(strategy="median")), *extra, ("sc", StandardScaler())])

    groups = [
        ("bmi", num([("clip", FunctionTransformer(clip_bmi))]), [c for c in ["BMI"] if c in f]),
        ("days", num(), days),
        ("ord", num(), [c for c in ORDINAL if c in f]),
        ("bin", SimpleImputer(strategy="most_frequent"), [c for c in BINARY if c in f]),
    ]
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", ColumnTransformer([g for g in groups if g[2]])),
        ("model", estimator),
    ])


# ----------------------------------------------------------- threshold
def find_threshold(y_true, proba, target_recall=0.80):
    """Highest probability threshold that still reaches `target_recall`.

    Screening cares about recall (missing a real case is worse than a false
    alarm), so we choose the threshold on validation data, never on the test set.
    """
    _, recall, thr = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])
