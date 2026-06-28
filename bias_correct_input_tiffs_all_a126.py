import os
import re
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# -----------------------------
# USER SETTINGS
# -----------------------------
for scenario in ['ssp126']:
    input_dir = f"/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_raw/ACCESS-CM2/{scenario}"
    bias_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/bias_correction_factors/"
    output_dir = f"/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/ACCESS-CM2/{scenario}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Map: original band name -> bias band name
    band_map = {
        "relative_humidity": "relative_humidity",
        "total_precipitation_sum": "total_precipitation_sum",
        #"rlds": "rlds",
        #"rsds": "rsds",
        #"sfcWind": "wsp",
        "temperature_2m": "temperature_2m",
        "temperature_2m_max": "temperature_2m_max",
        "temperature_2m_min": "temperature_2m_min",
        "build_up_index": "build_up_index",
        "drought_code": "drought_code",
        "duff_moisture_code": "duff_moisture_code",
        "fine_fuel_moisture_code": "fine_fuel_moisture_code",
        "fire_weather_index": "fire_weather_index",
        "initial_fire_spread_index": "initial_fire_spread_index"
    }
    
    BAND_LIMITS = {
        "relative_humidity": (0, 100),
        "total_precipitation_sum": (0, None),
        #"rlds": (0, None),
        #"rsds": (0, None),
        #"wsp": (0, None),
        "build_up_index": (0, None),
        "drought_code": (0, 3000), # OG max value is None
        "duff_moisture_code": (0, None),
        "fine_fuel_moisture_code": (0, None),
        "fire_weather_index": (0, None),
        "initial_fire_spread_index": (0, None),
    }
    
    # -----------------------------
    # HELPERS
    # -----------------------------
    def extract_month(filename):
        """Extract MM from ...YYYY_MM.tif"""
        m = re.search(r'_(\d{2})\.tif$', filename)
        return int(m.group(1)) if m else None
    
    
    def find_bias_file(month):
        pattern = os.path.join(bias_dir, f"bias_correction_factors_ACCESS-CM2_{scenario}_{month:02d}.tif")
        matches = glob.glob(pattern)
        return matches[0] if matches else None
    
    
    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    input_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    for in_file in input_files:
        fname = os.path.basename(in_file)
        month = extract_month(fname)
    
        if month is None:
            print(f"Skipping (no month found): {fname}")
            continue
    
        bias_file = find_bias_file(month)
        if bias_file is None:
            print(f"No bias file for month {month:02d}, skipping")
            continue
    
        print(f"Processing {fname} with bias file {os.path.basename(bias_file)}")
    
        with rasterio.open(in_file) as src, rasterio.open(bias_file) as bias_src:
    
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32)
            
            data = src.read().astype(np.float32)
            
            out_data = data.copy()
    
            src_band_names = list(src.descriptions)
            bias_band_names = list(bias_src.descriptions)
            
            # Build lookup: bias band name → index
            bias_lookup = {name: i+1 for i, name in enumerate(bias_band_names)}
    
            for i, band_name in enumerate(src_band_names):
            
                if band_name not in band_map:
                    print("Continuing for:", band_name)
                    continue
            
                var = band_map[band_name]
                
                # Build required band names
                mu_obs_name   = f"{var}_mu_obs"
                mu_model_name = f"{var}_mu_model"
                scale_name    = f"{var}_scale"
            
                # Check existence
                if not all(name in bias_lookup for name in [mu_obs_name, mu_model_name, scale_name]):
                    print(f"Missing bias bands for {var}")
                    continue
            
                # Get indices
                mu_obs_idx   = bias_lookup[mu_obs_name]
                mu_model_idx = bias_lookup[mu_model_name]
                scale_idx    = bias_lookup[scale_name]
            
                # Allocate arrays
                mu_obs_arr   = np.zeros((src.height, src.width), dtype=np.float32)
                mu_model_arr = np.zeros((src.height, src.width), dtype=np.float32)
                scale_arr    = np.zeros((src.height, src.width), dtype=np.float32)
            
                # Reproject all three
                for src_idx, dest_arr in zip(
                    [mu_obs_idx, mu_model_idx, scale_idx],
                    [mu_obs_arr, mu_model_arr, scale_arr]
                ):
                    reproject(
                        source=rasterio.band(bias_src, src_idx),
                        destination=dest_arr,
                        src_transform=bias_src.transform,
                        src_crs=bias_src.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        dst_shape=(src.height, src.width),
                        resampling=Resampling.nearest
                    )
    
                print(
                    var,
                    "mu_obs:",
                    np.nanmin(mu_obs_arr),
                    np.nanmax(mu_obs_arr),
                    "mu_model:",
                    np.nanmin(mu_model_arr),
                    np.nanmax(mu_model_arr),
                    "scale:",
                    np.nanmin(scale_arr),
                    np.nanmax(scale_arr)
                )
            
                X = data[i]
            
                # -----------------------------
                # FULL BIAS CORRECTION
                # -----------------------------
                corrected = mu_obs_arr + scale_arr * (X - mu_model_arr)
    
                print(
                    var,
                    np.nanmean(X),
                    np.nanmean(corrected),
                    np.nanmean(corrected - X)
                )
            
                # -----------------------------
                # Handle nodata
                # -----------------------------
                if src.nodata is not None:
                    mask = X == src.nodata
                else:
                    mask = np.isnan(X)
            
                # -----------------------------
                # Apply limits
                # -----------------------------
                if band_name in BAND_LIMITS:
                    min_val, max_val = BAND_LIMITS[band_name]
            
                    if min_val is not None:
                        corrected = np.maximum(corrected, min_val)
            
                    if max_val is not None:
                        corrected = np.minimum(corrected, max_val)
            
                corrected[mask] = src.nodata if src.nodata is not None else np.nan
            
                out_data[i] = corrected
    
            # -----------------------------
            # WRITE OUTPUT
            # -----------------------------
            out_path = os.path.join(output_dir, fname)
    
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_data.astype(np.float32))
    
                # Preserve original band names
                for i, desc in enumerate(src_band_names):
                    dst.set_band_description(i + 1, desc)
    
    print("Done.")