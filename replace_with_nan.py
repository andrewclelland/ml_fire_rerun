from pathlib import Path
import rasterio
import numpy as np
import tempfile
import os
import shutil

# -----------------------
# INPUTS
# -----------------------
input_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"

band_names = [
    "temperature_2m_max",
    "temperature_2m_min"
]

tif_files = list(Path(input_dir).rglob("*.tif"))

print(f"Found {len(tif_files)} files")

# -----------------------
# PROCESS FILES
# -----------------------
for tif_path in tif_files:
    tif_path = str(tif_path)

    print(f"Processing: {tif_path}")

    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")  # ensure NaN support
        profile = src.profile.copy()

        # map band names → indices
        band_map = {}
        for i in range(1, src.count + 1):
            name = src.descriptions[i - 1]
            if name in band_names:
                band_map[name] = i

        # replace -9999 with NaN
        for band_name, band_index in band_map.items():
            arr = data[band_index - 1]
            arr[arr == -9999] = np.nan
            data[band_index - 1] = arr

        profile.update(dtype="float32")

    # -----------------------
    # SAFE WRITE (TEMP FILE)
    # -----------------------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)

        # preserve band names
        dst.descriptions = tuple(src.descriptions)

    shutil.move(tmp_path, tif_path)
    
    print(f"Updated: {tif_path}")

print("Done.")