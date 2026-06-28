import os
from pathlib import Path
import rasterio
import numpy as np
import tempfile
import shutil

target_root = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"

band_names = [
    "build_up_index",
    "drought_code",
    "duff_moisture_code",
    "fine_fuel_moisture_code",
    "fire_weather_index",
    "initial_fire_spread_index"
]

# -----------------------
# LOOP OVER MONTHS
# -----------------------
for month in range(1, 13):

    source_tif = f"/home/users/clelland/Model/rerun/new_fwi/resampled_cems_fwi_new_cems_fwi_2023_{month}.tif"
    target_tif = f"{target_root}/observed_historic_2023_{month}.tif"

    print(f"\nProcessing month {month}")

    # -----------------------
    # OPEN SOURCE ONCE
    # -----------------------
    with rasterio.open(source_tif) as src:

        source_band_map = {}
        for band_name in band_names:
            band_index = None
            for i in range(1, src.count + 1):
                if src.descriptions[i - 1] == band_name:
                    band_index = i
                    break

            if band_index is None:
                raise ValueError(f"Band '{band_name}' not found in source tif")

            source_band_map[band_name] = src.read(band_index)

    # -----------------------
    # OPEN TARGET ONCE
    # -----------------------
    with rasterio.open(target_tif) as dst:
        profile = dst.profile
        data = dst.read()

        for band_name in band_names:

            # find target band index
            target_band_index = None
            for i in range(1, dst.count + 1):
                if dst.descriptions[i - 1] == band_name:
                    target_band_index = i
                    break

            if target_band_index is None:
                raise ValueError(f"Band '{band_name}' not found in target tif: {target_tif}")

            # shape check
            if data[target_band_index - 1].shape != source_band_map[band_name].shape:
                raise ValueError(
                    f"Shape mismatch for {band_name} in {target_tif}"
                )

            # replace band
            data[target_band_index - 1] = source_band_map[band_name]

        # -----------------------
        # WRITE ONCE (IMPORTANT)
        # -----------------------
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)

        profile.update(dtype=data.dtype)

        with rasterio.open(tmp_path, "w", **profile) as out:
            out.write(data)

    shutil.move(tmp_path, target_tif)
    print(f"Updated: {target_tif}")

print("\nDone.")