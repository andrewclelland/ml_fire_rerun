import os
import glob
import numpy as np
import rasterio

#root_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/ACCESS-CM2/ssp126"
root_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/new_training"

#tif_files = glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True)
tif_files = glob.glob(
    #os.path.join(root_dir, "**", "*2023*.tif"),
    os.path.join(root_dir, "*2023*.tif"),
    recursive=True
)

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