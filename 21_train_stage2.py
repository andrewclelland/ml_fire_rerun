#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NEEDS 125GB memory

"""
END-TO-END FINAL PIPELINE (TRAINING ONLY)
-- 1. Tunes XGBoost to find optimal trees and dynamic threshold --
-- 2. Trains Final XGBoost Model on 100% of data (Heavy Regularization) --
-- 3. Saves Model and Threshold Metrics for Later Inference --
"""

import os
import sys
import gc
import re
from pathlib import Path

# ============================================================
# GPU SELECTION: Force the script to only see and use GPU 0
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# --- INPUT PATHS ---
DATASET_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/new_training/parquet_historical")

# --- OUTPUT PATHS ---
# UPDATED: Changed output folder to reflect this is a final model, not LOYO
OUT_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/new_training/xgb_final_regularized_new_fwi")
MODELS_DIR = OUT_DIR / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LOG_FILE = OUT_DIR / "training_log_new_fwi_no_dc.txt"
SUMMARY_CSV_OUT = OUT_DIR / "final_metrics_summary_no_dc.csv"

# --- FEATURES & CONFIG ---
FEATURES = [
    "DEM", "slope", "aspect", "b1", "relative_humidity",
    "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", #"drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code",
    "fire_weather_index", "initial_fire_spread_index",
]

FRACTION_COL = "fraction"
LABEL_COL = "burned"
VAL_SIZE_FOR_TUNING = 0.20 

# TRAINING CONFIG
FINAL_ROUNDS  = 3000            
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 8
USE_GPU = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())

# ============================================================
# HELPERS
# ============================================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def find_area_match_threshold(y_true, y_probs):
    n_burned = np.sum(y_true)
    n_total = len(y_true)
    if n_burned == 0:
        return 0.99 
    target_percentile = 100.0 * (1.0 - (n_burned / n_total))
    return float(np.percentile(y_probs, target_percentile))

def prepare_df_cleaned(df: pd.DataFrame):
    df = df.copy()
    df[FRACTION_COL] = pd.to_numeric(df[FRACTION_COL], errors="coerce").astype("float32")
    df = df[df[FRACTION_COL].notna() & (df[FRACTION_COL] != 0.5)].copy()
    df[LABEL_COL] = (df[FRACTION_COL] > 0.5).astype("uint8")
    
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["b1"] = pd.to_numeric(df["b1"], errors="coerce").round().astype("Int64")

    #fwi_cols = ["duff_moisture_code", "drought_code", "fine_fuel_moisture_code", "build_up_index"]
    fwi_cols = ["duff_moisture_code", "fine_fuel_moisture_code", "build_up_index"]
    for c in fwi_cols:
        if c in df.columns:
            df = df[df[c] >= 0]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [LABEL_COL, "year", "month", "longitude", "latitude"])
    
    for c in FEATURES:
        if c == "b1": continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    
    df["b1"] = df["b1"].astype("category")
    return df

def load_data():
    print(f"Loading Dataset from: {DATASET_DIR}")
    dset = ds.dataset(str(DATASET_DIR), format="parquet", partitioning="hive")
    cols = FEATURES + [FRACTION_COL, "year", "month", "longitude", "latitude"]
    cols_to_load = [c for c in cols if c in dset.schema.names]
    table = dset.to_table(columns=cols_to_load)
    df = table.to_pandas()
    return prepare_df_cleaned(df)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    sys.stdout = Logger(str(LOG_FILE))
    print(f"Logging initialized. Writing to: {LOG_FILE}")
    print("-" * 50)
    print(f"Starting End-to-End Final Pipeline (GPU={USE_GPU})")
    
    df_all = load_data()
    print(f"Total Tabular Rows: {len(df_all):,}")
    
    X_all = df_all[FEATURES]
    y_all = df_all[LABEL_COL].values

    # --------------------------------------------------------
    # PHASE 1: TUNING (Find optimal trees & threshold)
    # --------------------------------------------------------
    print("\n" + "#" * 60)
    print("PHASE 1: TUNING FOR OPTIMAL TREES AND THRESHOLD")
    print("#" * 60)

    X_tune, X_val, y_tune, y_val = train_test_split(
        X_all, y_all, test_size=VAL_SIZE_FOR_TUNING, 
        random_state=RANDOM_STATE, stratify=y_all
    )

    n_pos_tune = y_tune.sum()
    n_neg_tune = len(y_tune) - n_pos_tune
    scale_weight_tune = n_neg_tune / max(1, n_pos_tune)

    dtrain_tune = xgb.DMatrix(X_tune, label=y_tune, nthread=N_JOBS, enable_categorical=True)
    dval_tune   = xgb.DMatrix(X_val,  label=y_val,  nthread=N_JOBS, enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "seed": RANDOM_STATE,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_weight_tune,
        "max_depth": 4,            
        "min_child_weight": 100,  
        "gamma": 5.0,              
        "subsample": 0.5,         
        "colsample_bytree": 0.5,  
        "reg_lambda": 10.0,       
        "reg_alpha": 1.0,         
    }

    print("Training tuning model with early stopping...")
    booster_tune = xgb.train(
        params=params,
        dtrain=dtrain_tune,
        num_boost_round=int(FINAL_ROUNDS),
        evals=[(dval_tune, "val")],
        early_stopping_rounds=200,
        verbose_eval=500 
    )

    # Calculate optimal boosting rounds and threshold
    best_iteration = booster_tune.best_iteration
    # XGBoost best_iteration is 0-indexed. Add 1 to get the actual number of trees
    optimal_trees = best_iteration + 1 
    
    val_probs = booster_tune.predict(dval_tune)
    final_thr = find_area_match_threshold(y_val, val_probs)
    
    print(f"> Optimal Number of Trees: {optimal_trees}")
    print(f"> Calculated Threshold: {final_thr:.4f}")

    # Clean up memory before full training
    del dtrain_tune, dval_tune, X_tune, X_val, y_tune, y_val, booster_tune
    gc.collect()

    # --------------------------------------------------------
    # PHASE 2: FINAL TRAINING ON 100% DATA
    # --------------------------------------------------------
    print("\n" + "#" * 60)
    print("PHASE 2: TRAINING FINAL MODEL ON 100% DATA")
    print("#" * 60)

    n_pos_all = y_all.sum()
    n_neg_all = len(y_all) - n_pos_all
    scale_weight_all = n_neg_all / max(1, n_pos_all)
    params["scale_pos_weight"] = scale_weight_all

    dtrain_all = xgb.DMatrix(X_all, label=y_all, nthread=N_JOBS, enable_categorical=True)

    print(f"Training final model with exact {optimal_trees} trees (no early stopping)...")
    final_booster = xgb.train(
        params=params,
        dtrain=dtrain_all,
        num_boost_round=optimal_trees,
        verbose_eval=500 
    )

    # Save Model
    model_name = "xgb_final_model_no_dc.json"
    save_path = MODELS_DIR / model_name
    final_booster.save_model(str(save_path))
    print(f"> Final model saved to: {save_path}")

    # ============================================================
    # FINAL SAVES & SUMMARIES
    # ============================================================
    print("\n" + "="*50)
    print("PIPELINE COMPLETE - SAVING SUMMARY")
    print("="*50)
    
    # Store metrics
    results = [{
        "total_obs_pixels": int(n_pos_all),
        "total_tabular_rows": len(df_all),
        "optimal_trees": optimal_trees,
        "final_threshold": float(final_thr)
    }]
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(SUMMARY_CSV_OUT, index=False)
    print(f"Saved comprehensive summary to: {SUMMARY_CSV_OUT}\n")
    
    print(df_results.to_string(index=False))
    print("-" * 50)

if __name__ == "__main__":
    main()