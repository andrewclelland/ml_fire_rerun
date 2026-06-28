import os
from pathlib import Path
import numpy as np
import rasterio

ROOT_DIR = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final/MRI-ESM2-0/"

TARGET_BANDS = [
    "total_precipitation_sum",
    #"temperature_2m",
    #"temperature_2m_max",
    #"temperature_2m_min",
]


def process_tif(tif_path):
    try:
        with rasterio.open(tif_path, "r+") as src:

            band_names = src.descriptions

            if not band_names:
                print(f"Skipping {tif_path}: no band descriptions found")
                return

            total_modified = 0

            for band_name in TARGET_BANDS:

                try:
                    band_idx = band_names.index(band_name) + 1
                except ValueError:
                    # Fallback to fixed band numbers if descriptions are missing
                    fallback = {
                        "total_precipitation_sum": 6,
                        "temperature_2m": 10,
                        "temperature_2m_max": 11,
                        "temperature_2m_min": 12,
                    }

                    band_idx = fallback.get(band_name)

                    if band_idx is None or band_idx > src.count:
                        print(f"  Missing band: {band_name}")
                        continue

                data = src.read(band_idx).astype(np.float32)

                #n_modified = np.sum(data == 0)
                n_modified = np.sum(data < 0)

                if n_modified > 0:
                    #data[data == 0] = np.nan
                    data[data < 0] = 0
                    src.write(data, band_idx)
                    total_modified += n_modified

                    print(
                        #f"  {band_name}: replaced {n_modified:,} zero values with NaN"
                        f"  {band_name}: replaced {n_modified:,} negative values with zero"
                    )

            if total_modified > 0:
                print(f"Updated {tif_path}: {total_modified:,} pixels modified")
            else:
                print(f"No changes needed: {tif_path}")

    except Exception as e:
        print(f"Error processing {tif_path}: {e}")

#tif_files = [
#    f for f in Path(ROOT_DIR).glob("*.tif") # for recursive use rglob
#    if "2100_" in f.name
#]


for tif_file in Path(ROOT_DIR).rglob("*.tif"):
#for tif_file in tif_files:
    process_tif(tif_file)

print("Done.")