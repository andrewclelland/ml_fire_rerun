#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1 LightGBM model (FASTER + FIXED for LightGBM 4.x):
Predict burned_label on 1-degree coarse-grid cells (EPSG:4326)

- Reads all *_grid1deg.parquet from 'new_fwi' folder across all years
- Uses selected predictor columns only
- Stratified K-Fold CV on a tuning subset to find optimal hyperparameters and tree count
- Trains final model on 100% of available data for inference
- Finds optimal probability threshold (0.10–0.90) using OOF from the tuning phase
- Saves model + metrics + plots to 'new_fwi' output folder
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import joblib

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PARQUET_DIR = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model"
    "/new_training/parquet_coarse_grids_annual_new_fwi"
)

OUT_DIR = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model"
    "/new_training/stage_1_model_new_fwi"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
N_ITER_SEARCH = 30

LGBM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8
EARLY_STOPPING_ROUNDS = 200
THRESHOLDS = np.arange(0.10, 0.91, 0.10)

# Tuning subset control
NEG_CAP_FOR_TUNING = 300_000

FEATURES = [
    "DEM",
    "slope",
    "aspect",
    "b1",
    "relative_humidity",
    "total_precipitation_sum",
    "temperature_2m",
    "temperature_2m_min",
    "temperature_2m_max",
    "build_up_index",
    #"drought_code",
    "duff_moisture_code",
    "fine_fuel_moisture_code",
    "fire_weather_index",
    "initial_fire_spread_index",
]
TARGET = "burned_label"


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def iou_from_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(np.uint8)
    return {
        "threshold": float(thr),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "iou": iou_from_confusion(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_all_grid1deg(parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*_grid1deg.parquet"))
    print(f"Looking in: {parquet_dir}")
    print(f"Found {len(files)} 1-degree parquet files")
    if not files:
        raise RuntimeError("No *_grid1deg.parquet files found")

    cols = FEATURES + [TARGET]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    return pd.concat(dfs, ignore_index=True)


def prepare_xy(df: pd.DataFrame):
    df = df[FEATURES + [TARGET]].dropna(axis=0).copy()
    df["b1"] = df["b1"].astype("Int64").astype("category")

    X = df[FEATURES]
    y = df[TARGET].astype(np.uint8).to_numpy()

    print("\nDataset size:", len(df))
    print("Class counts:", pd.Series(y).value_counts().to_dict())
    return X, y


def make_tuning_subset(X, y, neg_cap=NEG_CAP_FOR_TUNING, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, size=neg_cap, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)

    Xs = X.iloc[idx].copy()
    ys = y[idx].copy()

    print(
        f"\n[TUNING SUBSET] positives={len(pos_idx):,}, "
        f"negatives_used={len(neg_idx):,}, total={len(idx):,}"
    )
    return Xs, ys


# ---------------------------------------------------------------------
# CV TRAIN/EVAL WITH EARLY STOPPING (LightGBM 4.x callbacks)
# ---------------------------------------------------------------------
def cv_oof_prob_with_params(X, y, params, cv, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    """
    Train one param set across folds with early stopping and return:
      - mean recall across folds at default 0.5 threshold
      - OOF probabilities
      - average best iteration (used to set n_estimators for the final full-data model)
    """
    oof_prob = np.zeros(len(y), dtype=np.float32)
    fold_recalls = []
    fold_iters = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]

        model = LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            categorical_feature=["b1"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            ],
        )

        prob = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof_prob[va] = prob

        pred_05 = (prob >= 0.5).astype(np.uint8)
        fold_recalls.append(recall_score(y_va, pred_05, zero_division=0))

        best_iter = getattr(model, "best_iteration_", None) or model.n_estimators
        fold_iters.append(best_iter)
        print(f"  Fold {fold}: best_iter={best_iter} recall@0.5={fold_recalls[-1]:.4f}")

    return float(np.mean(fold_recalls)), oof_prob, int(np.mean(fold_iters))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_all_grid1deg(PARQUET_DIR)
    X, y = prepare_xy(df)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    print(f"Class imbalance neg/pos = {pos_weight:.1f}")
    print(f"Using LightGBM threads = {LGBM_THREADS}")
    print(f"LightGBM version = {lgb.__version__}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # --- tune on subset for speed ---
    X_tune, y_tune = make_tuning_subset(X, y)

    base_params = dict(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=LGBM_THREADS,
        verbosity=-1,
        n_estimators=10_000,  # Used only during tuning; early stopping intercepts this
    )

    param_dist = {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [31, 63, 127, 255],
        "max_depth": [-1, 5, 7, 9],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
        "scale_pos_weight": [pos_weight * f for f in [0.5, 1, 2, 4]],
    }

    sampler = list(ParameterSampler(param_dist, n_iter=N_ITER_SEARCH, random_state=RANDOM_STATE))

    print("\n[TUNING] Starting manual random search with early stopping")
    best_score = -1.0
    best_params = None
    best_n_estimators = 100
    best_oof_prob = None

    for i, p in enumerate(sampler, start=1):
        params = {**base_params, **p}
        print(f"\n  Config {i}/{N_ITER_SEARCH}: {p}")

        mean_recall, oof_prob, mean_iter = cv_oof_prob_with_params(
            X_tune, y_tune,
            params=params,
            cv=cv,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        print(f"  -> mean recall@0.5 (tuning subset): {mean_recall:.4f}")

        if mean_recall > best_score:
            best_score = mean_recall
            best_params = params
            best_n_estimators = mean_iter
            best_oof_prob = oof_prob

    if best_params is None:
        raise RuntimeError("Tuning failed to produce a best parameter set.")

    # Freeze the optimal number of trees found during tuning
    best_params["n_estimators"] = best_n_estimators
    print("\n[BEST PARAMS FOUND]")
    print(json.dumps(best_params, indent=2))

    # --- Train final model on 100% of the data ---
    print("\n[FINAL TRAINING] Fitting final model on 100% of data (No CV, No Early Stopping)...")
    final_model = LGBMClassifier(**best_params)
    final_model.fit(
        X, y,
        categorical_feature=["b1"]
    )

    model_path = OUT_DIR / "lgbm_stage1_model_no_dc.joblib"
    joblib.dump(final_model, model_path)

    # --- Find Optimal Threshold ---
    # We use the OOF probabilities from the *tuning subset* to evaluate thresholds. 
    # Evaluating thresholds on the 100% trained model would overfit.
    print("\n[THRESHOLDS] Computing metrics across thresholds using tuning OOF probabilities")
    rows = [metrics_at_threshold(y_tune, best_oof_prob, t) for t in THRESHOLDS]
    df_thr = pd.DataFrame(rows)

    best_row = (
        df_thr
        .sort_values(["recall", "precision", "f1"], ascending=False)
        .iloc[0]
    )

    df_thr.to_csv(OUT_DIR / "threshold_metrics_no_dc.csv", index=False)

    with open(OUT_DIR / "final_metrics_no_dc.txt", "w") as f:
        f.write("Stage-1 LightGBM (1° grid) - Final Model Tuning Stats\n")
        f.write(json.dumps(best_row.to_dict(), indent=2))

    plt.figure()
    plt.plot(df_thr["threshold"], df_thr["recall"], marker="o")
    plt.xlabel("Probability threshold")
    plt.ylabel("Recall")
    plt.title("Threshold vs Recall (Tuning Subset OOF)")
    plt.grid(True)
    plt.savefig(OUT_DIR / "threshold_vs_recall_no_dc.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n=== BEST THRESHOLD ===")
    print(best_row)

    print(f"\nArtifacts saved to:\n{OUT_DIR}")
    print(f"Model saved to:\n{model_path}")


if __name__ == "__main__":
    main()