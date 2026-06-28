import os
import glob
import rasterio

# root folder containing all output tifs
root_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/observed_historic_data/"
#root_dir = "/home/users/clelland/Model/rerun/new_fwi/"

# define your desired band names (must match band count!)
NEW_BAND_NAMES = [
    "DEM",
    "slope",
    "aspect",
    "b1",
    "relative_humidity",
    "total_precipitation_sum",
    #"longwave_radiation",
    #"shortwave_radiation",
    #"wind_speed",
    "temperature_2m",
    "temperature_2m_min",
    "temperature_2m_max",
    "build_up_index",
    "drought_code",
    "duff_moisture_code",
    "fine_fuel_moisture_code",
    "fire_weather_index",
    "initial_fire_spread_index",
    "fraction"
]

# find all tif files recursively
tif_files = glob.glob(os.path.join(root_dir, "**", "*2023*.tif"), recursive=True)

print(f"Found {len(tif_files)} files")

for tif in tif_files:

    #with rasterio.open(tif, "r+") as dst:
    with rasterio.open(tif, "r+", IGNORE_COG_LAYOUT_BREAK='YES') as dst:

        if dst.count != len(NEW_BAND_NAMES):
            print(f"Skipping {tif} (band mismatch: {dst.count})")
            continue

        for i, name in enumerate(NEW_BAND_NAMES, start=1):
            dst.set_band_description(i, name)

    print("Updated:", tif)