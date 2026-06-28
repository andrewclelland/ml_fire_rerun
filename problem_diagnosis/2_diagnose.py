import os
import re
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.dataset as pa_ds
import rasterio as rio
import rasterio.mask
import shap
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import warnings

# Robust check for PROJ database
import pyproj
if 'CONDA_PREFIX' in os.environ:
    proj_path = os.path.join(os.environ['CONDA_PREFIX'], 'share', 'proj')
    if os.path.exists(proj_path):
        os.environ['PROJ_LIB'] = proj_path
        pyproj.datadir.set_data_dir(proj_path)
    else:
        alt_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'proj')
        if os.path.exists(alt_path):
            os.environ['PROJ_LIB'] = alt_path
            pyproj.datadir.set_data_dir(alt_path)

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
# Directories for BOTH data sources
PARQUET_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/parquets_indiv/parquet_ACCESS-CM2_ssp126")
CEMS_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data")
MODEL_PATH = Path("/home/users/clelland/Model/rerun/xgb_final_model.json")
OUT_DIR = Path("/home/users/clelland/Model/rerun/problem_diagnosis/shapley_waterfalls")
BEST_CELL_PATH = Path("/home/users/clelland/Model/rerun/problem_diagnosis/best_match_5deg_cell_indiv.geojson")

TARGET_MONTH = 7 # Isolate peak summer conditions

# Define the scenarios to compare
SCENARIOS = [
    {"model": "ACCESS-CM2_ssp126", "year": 2026, "source": "parquet"},
    {"model": "ERA5", "year": 2025, "source": "tif"} 
]

FEATURES = [
    "DEM", "slope", "aspect", "b1", "relative_humidity",
    "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", "drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code",
    "fire_weather_index", "initial_fire_spread_index",
]

def norm_str(s): 
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def run_aoi_shap_explanations():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Explainer
    print("Loading XGBoost Model...")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    explainer = shap.TreeExplainer(bst)
    explainer.model._xgb_dmatrix_props = {"enable_categorical": True} # Prevent categorical crash

    # 2. Load Winning 5-Degree Cell
    print("Loading saved 5-degree winning cell...")
    if not BEST_CELL_PATH.exists():
        raise FileNotFoundError("Could not find the saved cell. Run the search script first.")
    roi_gdf = gpd.read_file(BEST_CELL_PATH)
    #roi_6931 = roi_gdf.to_crs("EPSG:6931")
    #aoi_poly_6931 = roi_6931.geometry.iloc[0]
    aoi_poly_6931 = roi_gdf.geometry.iloc[0]

    # 3. Process Scenarios
    for scenario in SCENARIOS:
        t_model = scenario["model"]
        t_year = scenario["year"]
        source = scenario["source"]
        
        print(f"\n{'='*50}")
        print(f"Processing Model: {t_model} | Year: {t_year} | Month: {TARGET_MONTH} | Source: {source.upper()}")
        print(f"{'='*50}")
        
        aoi_df = pd.DataFrame()

        # =================================================================
        # PIPELINE A: FUTURE (PARQUET)
        # =================================================================
        if source == "parquet":
            dataset = pa_ds.dataset(str(PARQUET_DIR), format="parquet", partitioning="hive")
            df = dataset.to_table(
                #filter=(pa_ds.field("model") == t_model) & (pa_ds.field("year") == t_year)
                filter=(pa_ds.field("year") == t_year)
            ).to_pandas()

            if df.empty:
                print(f"[!] No Parquet data found for {t_model} in {t_year}. Skipping.")
                continue
                
            # Filter for specific month
            if 'month' in df.columns:
                df = df[df['month'] == TARGET_MONTH]
                if df.empty:
                    print(f"[!] No Parquet data found for month {TARGET_MONTH}. Skipping.")
                    continue

            print("  Filtering pixels geographically...")
            gdf_points = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326"
            )            
            inside = gdf_points.within(aoi_poly_6931)
            aoi_df = df.loc[inside].copy()

            #if not aoi_df.empty:
                # Apply WIE-MIP specific unit conversions (to Kelvin for the model)
                #print("  Applying WIE-MIP unit conversions...")
                #aoi_df['temperature_2m'] += 273.15
                #aoi_df['temperature_2m_max'] += 273.15
                #aoi_df['temperature_2m_min'] += 273.15
                #aoi_df['total_precipitation_sum'] *= 2629.8

        # =================================================================
        # PIPELINE B: CEMS / ERA5 (GeoTIFFs)
        # =================================================================
        elif source == "tif":
            # Extract only the targeted month's TIF files using regex
            tifs = []
            for f in CEMS_DIR.glob(f"observed_historic_{t_year}_*.tif"):
                match = re.search(r"historic_\d{4}_(\d{1,2})\.tif$", f.name)
                if match and int(match.group(1)) == TARGET_MONTH:
                    tifs.append(f)
                    
            if not tifs:
                print(f"[!] No GeoTIFFs found for {t_year}-{TARGET_MONTH} in {CEMS_DIR}. Skipping.")
                continue
                
            target_map = {norm_str(f): f for f in FEATURES}
            target_map[norm_str("initial_spread_index")] = "initial_fire_spread_index"
            
            all_months_pixels = []
            for tif_path in sorted(tifs):
                with rio.open(tif_path) as src:
                    roi_proj = roi_gdf.to_crs(src.crs)
                    try:
                        out_img, _ = rasterio.mask.mask(src, roi_proj.geometry, crop=True, nodata=np.nan)
                    except ValueError:
                        continue # AOI doesn't overlap
                        
                    b_map = {target_map[norm_str(desc)]: i for i, desc in enumerate(src.descriptions) if desc and norm_str(desc) in target_map}
                    pixels = {std_name: out_img[idx].flatten() for std_name, idx in b_map.items()}
                    df_pix = pd.DataFrame(pixels).dropna()
                    if not df_pix.empty:
                        all_months_pixels.append(df_pix)
            
            if all_months_pixels:
                aoi_df = pd.concat(all_months_pixels)
                print("  Extracted raw CEMS TIFF data.")
                
                # Apply CEMS-specific scaling for precipitation
                #if "total_precipitation_sum" in aoi_df.columns:
                #    print("  Scaling CEMS/ERA5 precipitation (* 1,000,000)...")
                #    aoi_df["total_precipitation_sum"] *= 1000000

        # =================================================================
        # SHARED: AGGREGATION & SHAP
        # =================================================================
        if aoi_df.empty:
            print("[!] No valid pixels found for this scenario after extraction. Skipping.")
            continue

        print(f"  Aggregating {len(aoi_df)} pixels...")
        aoi_df = aoi_df.dropna(subset=['fire_weather_index'])

        numeric_df = aoi_df.drop(columns=['b1', 'geometry', 'month', 'year', 'model'], errors='ignore').mean(numeric_only=True).to_frame().T
        
        # Categorical handling
        mode_b1 = aoi_df['b1'].round().mode()[0] if 'b1' in aoi_df.columns else 1
        
        pixel_df = numeric_df.copy()
        pixel_df['b1'] = mode_b1
        pixel_df['b1'] = pixel_df['b1'].astype(float).astype(int).astype("category")

        # Ensure exact feature order for XGBoost
        for f in FEATURES:
            if f not in pixel_df.columns:
                pixel_df[f] = np.nan
        X_pixel = pixel_df[FEATURES].copy()

        # Generate SHAP
        print("  Generating SHAP values...")
        shap_values = explainer(X_pixel)

        # --- FAHRENHEIT CONVERSION FOR PLOTTING ---
        # Modify the display values inside the SHAP Explanation object 
        # (This changes the text on the plot without altering the underlying math)
        temp_features = ["temperature_2m", "temperature_2m_max", "temperature_2m_min"]
        
        # 1. Convert values in the data array from Kelvin to Fahrenheit
        #for i, col in enumerate(X_pixel.columns):
        #    if col in temp_features:
        #        k_val = shap_values[0].data[i]
        #        f_val = (k_val - 273.15) * 1.8 + 32
        #        shap_values[0].data[i] = f_val
                
        # 2. Update feature names to show (°F)
        #new_feature_names = [f"{col} (°F)" if col in temp_features else col for col in X_pixel.columns]
        new_feature_names = [f"{col} (K)" if col in temp_features else col for col in X_pixel.columns]
        shap_values.feature_names = new_feature_names
        # ------------------------------------------

        # Plot and Save
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.title(f"SHAP: Fire Probability | 5° AOI Mean ({t_model} - {t_year}-0{TARGET_MONTH})")

        png_name = f"SHAP_5deg_AOI_{t_model}_{t_year}_0{TARGET_MONTH}.png"
        save_path = OUT_DIR / png_name

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Plot saved to: {save_path}")
        
        plt.show()
        plt.close()

if __name__ == "__main__":
    run_aoi_shap_explanations()