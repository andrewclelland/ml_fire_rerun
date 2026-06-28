from pathlib import Path
import rasterio
import numpy as np

# -----------------------
# INPUT / OUTPUT
# -----------------------
input_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"
output_dir = "/home/users/clelland/Model/rerun/new_fwi"

Path(output_dir).mkdir(parents=True, exist_ok=True)

# bands 5–9 inclusive (1-based indexing in rasterio)
band_indices = list(range(5, 10))

# -----------------------
# LOOP OVER MONTHS
# -----------------------
for month in range(1, 13):

    in_file = f"{input_dir}/observed_historic_2023_{month}.tif"
    out_file = f"{output_dir}/new_e5l_2023_{month}.tif"

    print(f"Processing month {month}")

    with rasterio.open(in_file) as src:

        # read selected bands
        data = src.read(band_indices)

        # copy metadata and update for new band count
        profile = src.profile.copy()
        profile.update(
            count=len(band_indices),
            dtype=data.dtype
        )

        # preserve band descriptions if they exist
        if src.descriptions:
            profile_descriptions = [src.descriptions[i - 1] for i in band_indices]
        else:
            profile_descriptions = None

    # -----------------------
    # WRITE OUTPUT
    # -----------------------
    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(data)

        if profile_descriptions:
            dst.descriptions = tuple(profile_descriptions)

    print(f"Saved: {out_file}")

print("Done.")