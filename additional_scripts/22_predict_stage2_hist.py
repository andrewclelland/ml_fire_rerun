import os
import re
import gc
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio as rio
import xgboost as xgb
import sys

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
IN_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_hist/training_e5l_cems_firecci_new_fwi_with_fraction")
MODELS_DIR = Path("/home/users/clelland/Model/rerun")

MONTHLY_PROB_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/inference_outputs_indiv/monthly_probs_raw")
MONTHLY_PROB_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "DEM", "slope", "aspect", "b1", "relative_humidity",
    "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", #"drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code",
    "fire_weather_index", "initial_fire_spread_index",
]

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def sanitize_names(names):
    seen = {}
    out = []
    for n in names:
        n0 = re.sub(r"[^a-zA-Z0-9_]", "_", str(n).strip()).strip("_")
        if n0 in seen:
            seen[n0] += 1
            n0 = f"{n0}_{seen[n0]}"
        else:
            seen[n0] = 1
        out.append(n0)
    return out

# ----------------------------------------------------------------------
# INFERENCE LOOP
# ----------------------------------------------------------------------
all_tifs = sorted(IN_DIR.glob("cems_e5l_*.tif"))
model_cache = {}

total_files = len(all_tifs)
print(f"Starting inference on {total_files} files...")
print("-" * 50)

for i, tif_path in enumerate(all_tifs, 1):
    match = re.search(r"firecci_(\d{4})_(\d+)", tif_path.name)
    if not match:
        continue
    
    year, month = int(match.group(1)), int(match.group(2))
    output_file = MONTHLY_PROB_DIR / f"prob_{year}_{month:02d}.tif"
    
    # Check if already done
    if output_file.exists():
        continue

    # Load model for the specific year
    if year not in model_cache:
        model_path = MODELS_DIR / "xgb_final_model_no_dc.json"
        if not model_path.exists():
            print(f"[{i}/{total_files}] ⚠️ Missing Model. Skipping.")
            continue
        
        bst = xgb.Booster()
        try:
            bst.set_param({'device': 'cuda'}) 
        except:
            bst.set_param({'device': 'cpu'})
        bst.load_model(str(model_path))
        model_cache[year] = bst
        print("--- Loaded Model ---")
    
    bst = model_cache[year]

    # Raster Prediction
    with rio.open(tif_path) as src:
        out_meta = src.meta.copy()
        out_meta.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')

        data = src.read().astype(np.float32)
        raw_descriptions = src.descriptions if src.descriptions else [f"B{i}" for i in range(1, src.count+1)]
        band_names = sanitize_names(raw_descriptions)
        
        df_pixels = pd.DataFrame(data.reshape(src.count, -1).T, columns=band_names)
        
        if "b1" in df_pixels.columns:
            df_pixels["b1"] = df_pixels["b1"].round().astype("Int64").astype("category")

        valid_mask = df_pixels["build_up_index"].notna().values
        probs_flat = np.full(len(df_pixels), np.nan, dtype=np.float32)

        if valid_mask.any():
            dtest = xgb.DMatrix(df_pixels.loc[valid_mask, FEATURES], enable_categorical=True)
            probs_flat[valid_mask] = bst.predict(dtest)

        prob_raster = probs_flat.reshape(src.height, src.width)
        
        with rio.open(output_file, 'w', **out_meta) as dst:
            dst.write(prob_raster, 1)
        
        # Simple print statement
        print(f"[{i}/{total_files}] Done: {year}-{month:02d}")
        
        # Memory Cleanup
        del data, df_pixels, probs_flat, prob_raster, valid_mask
        gc.collect()

print("-" * 50)
print("✅ All monthly probability TIFFs generated.")