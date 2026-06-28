from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================

MONTH = 7

OBS_DIR = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"
)

FUTURE_ROOT = Path(
    "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv"
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
# READ TIFF BAND MEANS
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
# STORE RESULTS
# =============================================================================

records = []

# =============================================================================
# OBSERVED (2001-2025)
# =============================================================================
print("Started historical obs")

for year in range(2001, 2026):

    tif = OBS_DIR / f"observed_historic_{year}_{MONTH}.tif"

    if not tif.exists():
        print(f"Missing: {tif}")
        continue

    band_means = get_band_means(tif)

    for band, value in band_means.items():

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

            tif = scenario_dir / f"{model}_{scenario}_{year}_{MONTH:02d}.tif"

            if not tif.exists():
                continue

            band_means = get_band_means(tif)

            for band, value in band_means.items():

                records.append({
                    "year": year,
                    "band": band,
                    "value": value,
                    "series": series_name
                })

        print(model, scenario, "done")

# =============================================================================
# DATAFRAME
# =============================================================================

df = pd.DataFrame(records)

print("df shape: ", df.shape)

# =============================================================================
# PLOT EACH BAND
# =============================================================================

outdir = Path("band_timeseries")
outdir.mkdir(exist_ok=True)

for band in sorted(df.band.unique()):

    plt.figure(figsize=(12, 6))

    subset = df[df.band == band]

    for series, group in subset.groupby("series"):

        group = group.sort_values("year")

        plt.plot(
            group["year"],
            group["value"],
            label=series,
            linewidth=2
        )

    plt.title(f"{band} (Month {MONTH})")
    plt.xlabel("Year")
    plt.ylabel("Spatial Mean")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        outdir / f"./ecoregion/{band}_month_{MONTH}.png",
        dpi=300
    )

    plt.close()

print("Finished.")