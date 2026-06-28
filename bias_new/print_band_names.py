import os
import rasterio

tif_folder = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data"

for filename in sorted(os.listdir(tif_folder)):
    if filename.lower().endswith((".tif", ".tiff")):
        tif_path = os.path.join(tif_folder, filename)

        print(f"\n{filename}")

        with rasterio.open(tif_path) as src:
            for band_idx in range(1, src.count + 1):
                band_name = src.descriptions[band_idx - 1]

                if band_name is None:
                    band_name = f"Band {band_idx}"

                print(f"  {band_idx}: {band_name}")