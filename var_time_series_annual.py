from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

# =============================================================================
# CONFIG
# =============================================================================

OBS_DIR = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"
)

FUTURE_ROOT = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_raw"
)

MODELS = [
    "ACCESS-CM2",
    "MRI-ESM2-0"
]

SCENARIOS = [
    "ssp126",
    "ssp245",
    "ssp370"
]

# =============================================================================
# READ SPATIAL BAND MEANS FROM A TIFF
# =============================================================================

def get_band_means(tif_file):

    with rasterio.open(tif_file) as src:

        band_names = src.descriptions

        if band_names is None or all(x is None for x in band_names):
            band_names = [f"band_{i+1}" for i in range(src.count)]

        means = {}

        for i, band in enumerate(band_names, start=1):

            arr = src.read(i).astype(np.float32)

            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan

            means[band] = np.nanmean(arr)

    return means


# =============================================================================
# ANNUAL MEAN OF MONTHLY SPATIAL MEANS
# =============================================================================

def get_yearly_band_means(monthly_tifs):

    monthly_values = {}

    for tif in monthly_tifs:

        if not tif.exists():
            continue

        band_means = get_band_means(tif)

        for band, value in band_means.items():
            monthly_values.setdefault(band, []).append(value)

    return {
        band: np.nanmean(values)
        for band, values in monthly_values.items()
    }


# =============================================================================
# STORE RESULTS
# =============================================================================

records = []

# =============================================================================
# OBSERVED (2001-2025)
# =============================================================================

print("Started historical obs")

for year in range(2001, 2026):

    monthly_tifs = [
        OBS_DIR / f"observed_historic_{year}_{month}.tif"
        for month in range(1, 13)
    ]

    yearly_means = get_yearly_band_means(monthly_tifs)

    for band, value in yearly_means.items():

        records.append({
            "year": year,
            "band": band,
            "value": value,
            "series": "Observed"
        })

print("Historical done")

# =============================================================================
# FUTURES (2026-2100)
# =============================================================================

for model in MODELS:

    for scenario in SCENARIOS:

        scenario_dir = FUTURE_ROOT / model / scenario

        if not scenario_dir.exists():
            print(f"Missing directory: {scenario_dir}")
            continue

        series_name = f"{model}_{scenario}"

        for year in range(2026, 2101):

            monthly_tifs = [
                scenario_dir / f"{model}_{scenario}_{year}_{month:02d}.tif"
                for month in range(1, 13)
            ]

            yearly_means = get_yearly_band_means(monthly_tifs)

            for band, value in yearly_means.items():

                records.append({
                    "year": year,
                    "band": band,
                    "value": value,
                    "series": series_name
                })

        print(model, scenario, "done")

# =============================================================================
# SAVE TO CSV
# =============================================================================

df = pd.DataFrame(records)

print("df shape:", df.shape)

csv_path = "yearly_band_means_raw.csv"
df.to_csv(csv_path, index=False)

print(f"Saved: {csv_path}")