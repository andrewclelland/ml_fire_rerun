import os
from pathlib import Path
import numpy as np
import rasterio

ROOT_DIR = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final/MRI-ESM2-0/ssp370"


def process_tif(tif_path):
    try:
        with rasterio.open(tif_path, "r+") as src:

            # Get band names/descriptions
            band_names = src.descriptions

            if not band_names:
                print(f"Skipping {tif_path}: no band descriptions found")
                return

            try:
                band_idx = band_names.index("relative_humidity") + 1
            except ValueError:
                print(f"Skipping {tif_path}: no 'relative_humidity' band")
                return

            data = src.read(band_idx)

            n_modified = np.sum(data > 100)

            if n_modified > 0:
                data = np.minimum(data, 100)
                src.write(data, band_idx)
                print(f"Updated {tif_path}: capped {n_modified:,} pixels")
            else:
                print(f"No changes needed: {tif_path}")

    except Exception as e:
        print(f"Error processing {tif_path}: {e}")


for tif_file in Path(ROOT_DIR).rglob("*.tif"):
    process_tif(tif_file)

print("Done.")