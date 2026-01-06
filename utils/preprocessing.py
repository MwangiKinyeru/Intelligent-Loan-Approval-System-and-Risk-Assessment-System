import pandas as pd
import numpy as np
import pickle
import os

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "cluster_info.pkl"), "rb") as f:
    cluster_info = pickle.load(f)

TRAINING_COLUMNS = cluster_info["feature_names"]

LOG_COLS = ["income", "dti_ratio"]
SCALE_COLS = [
    "credit_score",
    "income",
    "dti_ratio",
    "credit_history_length"
]


def _base_preprocess(data):
    df = pd.DataFrame([data])

    df = pd.get_dummies(
        df,
        columns=["home_ownership", "loan_purpose"],
        drop_first=True
    )

    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[TRAINING_COLUMNS]

    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def preprocess_for_classification(data, scaler):
    df = _base_preprocess(data)
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


def preprocess_for_regression(data, scaler):
    df = _base_preprocess(data)
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df
