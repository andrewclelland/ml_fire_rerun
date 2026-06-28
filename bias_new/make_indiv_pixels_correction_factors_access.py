import os
import re
import glob
import gc
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from collections import defaultdict

# ============================================================
# USER INPUTS
# ============================================================

obs_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"

historic_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/inputs_raw/ACCESS-CM2/historical"

scenario_dirs = {
    "ssp126": "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/inputs_raw/ACCESS-CM2/ssp126",
    "ssp245": "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/inputs_raw/ACCESS-CM2/ssp245",
    "ssp370": "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/inputs_raw/ACCESS-CM2/ssp370"
}

output_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/bias_correction_factors"
os.makedirs(output_dir, exist_ok=True)

variables_to_correct = [
    "relative_humidity", "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", "drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code", "fire_weather_index", "initial_fire_spread_index"
]

# Observed raster band mapping
obs_band_map = {
    "relative_humidity": 5,
    "total_precipitation_sum": 6,
    "temperature_2m": 7,
    "temperature_2m_min": 8,
    "temperature_2m_max": 9,
    "build_up_index": 11,
    "drought_code": 12,
    "duff_moisture_code": 13,
    "fine_fuel_moisture_code": 14,
    "fire_weather_index": 15,
    "initial_fire_spread_index": 16
}

# CMIP/model raster band mapping
model_band_map = {
    "relative_humidity": 5,
    "total_precipitation_sum": 6,
    "temperature_2m": 10,
    "temperature_2m_min": 12,
    "temperature_2m_max": 11,
    "build_up_index": 13,
    "drought_code": 14,
    "duff_moisture_code": 15,
    "fine_fuel_moisture_code": 16,
    "fire_weather_index": 17,
    "initial_fire_spread_index": 18
}

BAND_LIMITS = {
    "relative_humidity": (0, 100),
    "build_up_index": (0, None),
    "drought_code": (0, 3000),
    "duff_moisture_code": (0, None),
    "fine_fuel_moisture_code": (0, None),
    "fire_weather_index": (0, None),
    "initial_fire_spread_index": (0, None),
}


# Chunk size for memory efficiency
BLOCK_SIZE = 512


# ============================================================
# HELPERS
# ============================================================

def parse_date(filepath):
    """
    Extract year and month:
    """
    fname = os.path.basename(filepath)

    match = re.search(r"_(\d{4})_(\d{1,2})\.tif$", fname)

    if not match:
        raise ValueError(f"Could not parse date: {fname}")

    year = int(match.group(1))
    month = int(match.group(2))

    return year, month


def group_files_by_month(folder, year_min=None, year_max=None):
    """
    Returns:
    {month: [files]}
    """
    files = glob.glob(os.path.join(folder, "*.tif"))

    monthly = defaultdict(list)

    for fp in files:

        year, month = parse_date(fp)

        if year_min and year < year_min:
            continue
        if year_max and year > year_max:
            continue

        monthly[month].append(fp)

    for m in monthly:
        monthly[m] = sorted(monthly[m])

    return monthly


def safe_stats(arr):
    """
    Compute nan-safe mean/std across time axis.
    """
    mu = np.nanmean(arr, axis=0)
    sigma = np.nanstd(arr, axis=0)

    sigma = np.nan_to_num(sigma, nan=0.0)

    return mu, sigma


def apply_band_limits(arr, var_name):
    """
    Replace impossible values with NaN using
    predefined variable limits.
    """

    if var_name not in BAND_LIMITS:
        return arr

    lower, upper = BAND_LIMITS[var_name]

    # Lower bound
    if lower is not None:
        arr = np.where(arr < lower, np.nan, arr)

    # Upper bound
    if upper is not None:
        arr = np.where(arr > upper, np.nan, arr)

    return arr


# ============================================================
# LOAD OBSERVED MONTHLY GROUPS
# ============================================================

print("Grouping observed files...")

obs_monthly = group_files_by_month(
    obs_dir,
    year_min=2001,
    year_max=2025
)

# ============================================================
# MAIN LOOP
# ============================================================

for scenario, scenario_dir in scenario_dirs.items():

    print(f"\nProcessing {scenario}")

    # Historic: 2001–2014
    hist_monthly = group_files_by_month(
        historic_dir,
        year_min=2001,
        year_max=2014
    )

    # Scenario: 2015–2025
    scen_monthly = group_files_by_month(
        scenario_dir,
        year_min=2015,
        year_max=2025
    )

    for month in range(1, 13):

        print(f"  Month {month:02d}")

        obs_files = obs_monthly[month]
        hist_files = hist_monthly[month]
        scen_files = scen_monthly[month]

        model_files = hist_files + scen_files

        if len(obs_files) == 0 or len(model_files) == 0:
            print(f"    Skipping month {month}")
            continue

        # Use template raster
        with rio.open(obs_files[0]) as src_template:

            profile = src_template.profile.copy()

            height = src_template.height
            width = src_template.width
            transform = src_template.transform

            out_band_count = len(variables_to_correct) * 3
            
            profile.update(
                count=out_band_count,
                dtype="float32",
                compress="lzw",
                tiled=True,
                BIGTIFF="YES"
            )

            output_path = os.path.join(
                output_dir,
                f"bias_correction_factors_ACCESS-CM2_{scenario}_{month:02d}.tif"
            )

            with rio.open(output_path, "w", **profile) as dst:

                band_names = []
                
                for var in variables_to_correct:
                    band_names.extend([
                        f"{var}_mu_obs",
                        f"{var}_mu_model",
                        f"{var}_scale"
                    ])
                
                dst.descriptions = tuple(band_names)

                # Process in chunks/windows
                for row in range(0, height, BLOCK_SIZE):

                    for col in range(0, width, BLOCK_SIZE):

                        win_height = min(BLOCK_SIZE, height - row)
                        win_width = min(BLOCK_SIZE, width - col)

                        window = Window(
                            col_off=col,
                            row_off=row,
                            width=win_width,
                            height=win_height
                        )

                        output_stack = []

                        # ====================================================
                        # PROCESS EACH VARIABLE/BAND
                        # ====================================================

                        for var in variables_to_correct:
                        
                            obs_band = obs_band_map[var]
                            model_band = model_band_map[var]  
                            
                            # ---------------------------------
                            # Observed stack
                            # ---------------------------------
                            obs_stack = []
                        
                            for fp in obs_files:
                                with rio.open(fp) as src:
                        
                                    arr = src.read(
                                        obs_band,
                                        window=window
                                    ).astype(np.float32)

                                    # Convert nodata to NaN
                                    nodata = src.nodata

                                    if nodata is not None:
                                        arr = np.where(arr == nodata, np.nan, arr)

                                    # Apply physical limits
                                    arr = apply_band_limits(arr, var)
                        
                                    obs_stack.append(arr)
                        
                            obs_stack = np.stack(obs_stack)
                        
                            mu_obs, std_obs = safe_stats(obs_stack)
                        
                            del obs_stack
                        
                            # ---------------------------------v
                            # Model stack
                            # ---------------------------------
                            model_stack = []
                        
                            for fp in model_files:
                                with rio.open(fp) as src:
                        
                                    arr = src.read(
                                        model_band,
                                        window=window
                                    ).astype(np.float32)

                                    # Convert nodata to NaN
                                    nodata = src.nodata

                                    if nodata is not None:
                                        arr = np.where(arr == nodata, np.nan, arr)

                                    # Apply physical limits
                                    arr = apply_band_limits(arr, var)
                        
                                    model_stack.append(arr)
                        
                            model_stack = np.stack(model_stack)
                        
                            mu_model, std_model = safe_stats(model_stack)
                        
                            del model_stack
                        
                            # ---------------------------------
                            # Scale factor
                            # ---------------------------------
                            scale = np.divide(
                                std_obs,
                                std_model,
                                out=np.ones_like(std_obs),
                                where=(std_model > 0)
                            )
                        
                            # Failsafe
                            #scale = np.nan_to_num(scale, nan=0.0)
                        
                            output_stack.extend([
                                mu_obs.astype(np.float32),
                                mu_model.astype(np.float32),
                                scale.astype(np.float32)
                            ])
                        
                            gc.collect()

                        # =======================================
                        # Write chunk
                        # =======================================

                        output_stack = np.stack(output_stack)

                        dst.write(
                            output_stack,
                            window=window
                        )

        print(f"    Saved: {output_path}")

print("\nDone.")