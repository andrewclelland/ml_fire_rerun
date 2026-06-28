import os
import re
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import rasterio.mask
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import box

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
CEMS_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data")
WIEMIP_BASE_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final")
ROI_SHP_PATH = Path("/home/users/clelland/Model/rerun/problem_diagnosis/shapefile/resolve_shapefile_from_gee.shp")

OUT_PLOT = Path("/home/users/clelland/Model/rerun/problem_diagnosis/covariate_time_series_comparison_eco_indivcellmatch.png")
BEST_CELL_OUT = Path("/home/users/clelland/Model/rerun/problem_diagnosis/best_match_5deg_cell_indiv.geojson")
best_cell_id = 165

MODELS = ['ACCESS-CM2', 'MRI-ESM2-0']
SCENARIOS = ['ssp126', 'ssp245', 'ssp370']

FEATURES = [
    "relative_humidity", "total_precipitation_sum", 
    "temperature_2m", "temperature_2m_min", "temperature_2m_max", 
    "build_up_index", "drought_code", "duff_moisture_code", 
    "fine_fuel_moisture_code", "fire_weather_index", "initial_fire_spread_index"
]

FWI_COLS = [
    "build_up_index", "drought_code", "duff_moisture_code", 
    "fine_fuel_moisture_code", "fire_weather_index", "initial_fire_spread_index"
]

def norm_str(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

# =====================================================================
# 2. HELPER: DATA CLEANING & AGGREGATION
# =====================================================================
def clean_and_mean_pixels(df, is_wiemip=False):
    """Applies physical bounds and outlier removal to a dataframe of spatial pixels, returning the mean."""
    df = df.copy()
            
    # --- 2. PHYSICAL BOUNDS FILTER (Remove Fill Values / Artifacts) ---
    # Filtering in Kelvin
    for c in ["temperature_2m", "temperature_2m_min", "temperature_2m_max"]:
        if c in df.columns: df = df[(df[c] > 150) & (df[c] < 400)]
            
    if "total_precipitation_sum" in df.columns:
        df = df[df["total_precipitation_sum"] >= 0]
        #df["total_precipitation_sum"] * 1000
        
    for c in FWI_COLS:
        if c in df.columns: df = df[df[c] >= 0]
            
    # --- 3. OUTLIER REMOVAL (Statistical tails) ---
    cols_to_clean = [c for c in FEATURES if c != "relative_humidity" and c in df.columns]
    for c in cols_to_clean:
        q_low = df[c].quantile(0.005)
        q_high = df[c].quantile(0.995)
        df = df[(df[c] >= q_low) & (df[c] <= q_high)]
        
    # Return spatial mean of the clean pixels
    return df[FEATURES].mean().to_dict() if not df.empty else None


# =====================================================================
# 3. LOAD PRE-PROCESSED BEST CELL
# =====================================================================
roi_wgs84 = gpd.read_file(BEST_CELL_OUT).to_crs("EPSG:4326")

# CEMS files needed later in Section 4
tifs = [
    f for f in CEMS_DIR.glob("*.tif")
    if (match := re.search(r"observed_historic_(\d{4})_(\d{1,2})", f.name))
    and (2001 <= int(match.group(1)) <= 2025)
]

print(f"Loaded pre-selected cell {best_cell_id}")
print(f"Found {len(tifs)} CEMS monthly files")


# =====================================================================
# 4. EXTRACT CEMS / ERA5 (FROM GEOTIFFs)
# =====================================================================
print(f"\nProcessing CEMS/ERA5 GeoTIFFs (2001-2025) for Best Cell (ID: {best_cell_id})...")

target_map = {norm_str(f): f for f in FEATURES}
target_map[norm_str("initial_spread_index")] = "initial_fire_spread_index" 

cems_yearly_data = {}

for tif_path in tqdm(tifs, desc="Reading CEMS Monthly TIFs"):
    match = re.search(r"observed_historic_(\d{4})_(\d{1,2})", tif_path.name)
    year = int(match.group(1))
    
    with rio.open(tif_path) as src:
        roi_proj = roi_wgs84.to_crs(src.crs)
        
        try:
            out_img, _ = rasterio.mask.mask(src, roi_proj.geometry, crop=True, nodata=np.nan)
        except ValueError:
            continue
            
        b_map = {}
        for i, desc in enumerate(src.descriptions):
            n_desc = norm_str(desc)
            if n_desc in target_map:
                b_map[target_map[n_desc]] = i
                
        pixels = {std_name: out_img[idx].flatten() for std_name, idx in b_map.items()}
        df_pixels = pd.DataFrame(pixels).dropna()
        
        if not df_pixels.empty:
            clean_means = clean_and_mean_pixels(df_pixels, is_wiemip=False)
            if clean_means:
                if year not in cems_yearly_data:
                    cems_yearly_data[year] = []
                cems_yearly_data[year].append(clean_means)

cems_results = []
for year, monthly_means in cems_yearly_data.items():
    df_year = pd.DataFrame(monthly_means)
    year_mean = df_year.mean().to_dict()
    year_mean['Year'] = year
    year_mean['Source'] = "Training (ERA5/CEMS)"
    cems_results.append(year_mean)

# Convert to DataFrame
cems_results = pd.DataFrame(cems_results)

if "total_precipitation_sum" in cems_results.columns:
    cems_results["total_precipitation_sum"] = cems_results["total_precipitation_sum"] / 1000000


# =====================================================================
# 5. EXTRACT FUTURE DATA (GeoTIFFs)
# =====================================================================
print("\nProcessing FUTURE GeoTIFFs (2026–2100)...")

future_results = []

for model in MODELS:
    for scenario in SCENARIOS:
        print(f"\nProcessing {model} - {scenario}...")

        folder = WIEMIP_BASE_DIR / model / scenario
        if not folder.exists():
            continue

        yearly_data = {}

        tifs = [
            f for f in folder.glob("*.tif")
            if (match := re.search(r"(\d{4})_(\d{1,2})", f.name))
            and (2026 <= int(match.group(1)) <= 2100)
        ]

        for tif_path in tqdm(tifs, desc=f"{model}-{scenario}"):

            match = re.search(r"(\d{4})_(\d{1,2})", tif_path.name)
            year = int(match.group(1))

            with rio.open(tif_path) as src:
                roi_proj = roi_wgs84.to_crs(src.crs)

                try:
                    out_img, _ = rasterio.mask.mask(
                        src, roi_proj.geometry, crop=True, nodata=np.nan
                    )
                except ValueError:
                    continue

                b_map = {}
                for i, desc in enumerate(src.descriptions):
                    n_desc = norm_str(desc)
                    if n_desc in target_map:
                        b_map[target_map[n_desc]] = i

                pixels = {
                    std_name: out_img[idx].flatten()
                    for std_name, idx in b_map.items()
                }

                df_pixels = pd.DataFrame(pixels).dropna()

                if not df_pixels.empty:
                    clean_means = clean_and_mean_pixels(df_pixels)

                    if clean_means:
                        yearly_data.setdefault(year, []).append(clean_means)

        for year, vals in yearly_data.items():
            df_year = pd.DataFrame(vals)
            year_mean = df_year.mean().to_dict()
            year_mean['Year'] = year
            year_mean['Source'] = f"{model} ({scenario})"
            future_results.append(year_mean)


# =====================================================================
# 6. COMBINE, CONVERT, & PLOT
# =====================================================================
print("\nCombining datasets...")
df_combined = pd.concat(
    [cems_results, pd.DataFrame(future_results)],
    ignore_index=True
)

# --- POST-PROCESSING UNIT CONVERSIONS (Kelvin to Fahrenheit) ---
temp_features = ["temperature_2m", "temperature_2m_min", "temperature_2m_max"]

print("Generating Time Series plots...")
fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=False)
axes = axes.flatten()

palette = {
    "Training (ERA5/CEMS)": "black",
    "ACCESS-CM2 (ssp126)": "lightblue",     
    "ACCESS-CM2 (ssp245)": "blue",     
    "ACCESS-CM2 (ssp370)": "darkblue",     
    "MRI-ESM2-0 (ssp126)": "lightsalmon",  
    "MRI-ESM2-0 (ssp245)": "red",  
    "MRI-ESM2-0 (ssp370)": "brown"  
}

for i, feature in enumerate(FEATURES):
    ax = axes[i]
    for source, color in palette.items():
        df_sub = df_combined[df_combined['Source'] == source].sort_values('Year')
        
        if not df_sub.empty:
            linestyle = '-' if source == "Training (ERA5/CEMS)" else '--'
            marker = 'o' if source == "Training (ERA5/CEMS)" else ''
            
            ax.plot(
                df_sub['Year'], df_sub[feature], 
                color=color, 
                linewidth=2.5 if source == "Training (ERA5/CEMS)" else 2,
                linestyle=linestyle, marker=marker, markersize=4, label=source
            )
            
    # Add units to the specific plot titles
    plot_title = feature
    if feature in temp_features:
        plot_title = f"{feature} (K)"
    elif feature == "total_precipitation_sum":
        plot_title = f"{feature} (m)"
        
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Annual Spatial Mean")
    ax.grid(True, linestyle=':', alpha=0.7)
    if i == 0:
        ax.legend(loc='best', fontsize=10)

# Clean up empty subplots
for j in range(len(FEATURES), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f"Annual Mean Covariates (Best 5° Cell Match: {best_cell_id}): ERA5/CEMS vs FUTURE", fontsize=24, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {OUT_PLOT}")

plt.show()
plt.close()