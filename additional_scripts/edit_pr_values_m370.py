import os
import glob
import numpy as np
import rasterio
from pathlib import Path
import time

time.sleep(7200)

root_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/MRI-ESM2-0/ssp370"

tif_files = glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True)
#tif_files = [
#    f for f in Path(root_dir).glob("*.tif") # for recursive use rglob
#    if "2093_1" in f.name
#]

print(f"Found {len(tif_files)} files")

target_name = "total_precipitation_sum"
#target_name = "pr"

for tif in tif_files:

    with rasterio.open(tif, "r+") as dst:

        band_index = None

        for i, name in enumerate(dst.descriptions):
            if name == target_name:
                band_index = i + 1
                break

        if band_index is None:
            print(f"Skipping {tif} (band not found)")
            continue

        data = dst.read(band_index).astype("float32")
        data = data / 1000000

        dst.write(data, band_index)

    print("Updated:", tif)