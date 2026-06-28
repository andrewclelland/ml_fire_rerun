import os
import tempfile
import shutil

import numpy as np
import rasterio

tif_folder = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"

years = ("2023", "2024", "2025")

for filename in sorted(os.listdir(tif_folder)):
    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    if not any(year in filename for year in years):
        continue

    tif_path = os.path.join(tif_folder, filename)

    print(f"Processing {filename}")

    with rasterio.open(tif_path) as src:
        profile = src.profile
        data = src.read()  # shape: (bands, rows, cols)
        descriptions = list(src.descriptions)

    if data.shape[0] < 16:
        print(f"  Skipping: only {data.shape[0]} bands")
        continue

    # Convert to 0-based indices
    reordered = data.copy()

    # Save original bands 10-16
    b10 = data[9].copy()
    b11 = data[10].copy()
    b12 = data[11].copy()
    b13 = data[12].copy()
    b14 = data[13].copy()
    b15 = data[14].copy()
    b16 = data[15].copy()

    # Apply shift
    reordered[9]  = b16  # 16 -> 10
    reordered[10] = b10  # 10 -> 11
    reordered[11] = b11  # 11 -> 12
    reordered[12] = b12  # 12 -> 13
    reordered[13] = b13  # 13 -> 14
    reordered[14] = b14  # 14 -> 15
    reordered[15] = b15  # 15 -> 16

    # Reorder descriptions too
    desc = descriptions.copy()
    desc[9]  = descriptions[15]
    desc[10] = descriptions[9]
    desc[11] = descriptions[10]
    desc[12] = descriptions[11]
    desc[13] = descriptions[12]
    desc[14] = descriptions[13]
    desc[15] = descriptions[14]

    # Write to temporary file first
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(reordered)

        for i, d in enumerate(desc, start=1):
            if d is not None:
                dst.set_band_description(i, d)

    shutil.move(tmp_path, tif_path)

    print("  Done")