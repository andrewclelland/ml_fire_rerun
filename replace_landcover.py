import os
from pathlib import Path
import rasterio
import numpy as np
import tempfile
import shutil

# -----------------------
# INPUTS
# -----------------------
target_root = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_raw" # folders containing many tifs
source_tif = "/home/users/clelland/Model/rerun/baseland.tif" # tif containing replacement band

band_name = "b1"

# -----------------------
# LOAD SOURCE BAND
# -----------------------
with rasterio.open(source_tif) as src:
    # Find band index by name (GDAL band description)
    band_index = None
    for i in range(1, src.count + 1):
        if src.descriptions[i - 1] == band_name:
            band_index = i
            break

    if band_index is None:
        raise ValueError(f"Band '{band_name}' not found in source tif")

    replacement_band = src.read(band_index)

# -----------------------
# FIND ALL TARGET TIFFS
# -----------------------
tif_files = list(Path(target_root).rglob("*.tif"))

print(f"Found {len(tif_files)} files")

# -----------------------
# PROCESS EACH FILE
# -----------------------
for tif_path in tif_files:
    tif_path = str(tif_path)

    with rasterio.open(tif_path) as dst:
        profile = dst.profile
        data = dst.read()

        # Optional: verify dimensions match
        if data.shape[1:] != replacement_band.shape:
            print(f"Skipping (shape mismatch): {tif_path}")
            continue

        # Identify target band index (same name assumption)
        target_band_index = None
        for i in range(1, dst.count + 1):
            if dst.descriptions[i - 1] == band_name:
                target_band_index = i
                break

        if target_band_index is None:
            print(f"Skipping (band not found): {tif_path}")
            continue

        # Replace band
        data[target_band_index - 1, :, :] = replacement_band

        # Write to temp file first (safe overwrite)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)

        profile.update(dtype=data.dtype)

        with rasterio.open(tmp_path, "w", **profile) as out:
            out.write(data)

    shutil.move(tmp_path, tif_path)
    print(f"Updated: {tif_path}")

print("Done.")