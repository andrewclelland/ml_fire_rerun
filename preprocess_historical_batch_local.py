import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

roots = ['access', 'mri']
models = ["ACCESS-CM2", "MRI-ESM2-0"]
scenarios = ["historical"]
years = range(2001, 2015)
months = range(1, 13)

baseland_file = "/home/users/clelland/Model/rerun/baseland.tif"
tem_file = "/home/users/clelland/Model/rerun/TEM_map_grouped.tif"

cmip_root = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/raw_historic_data/cmip6"
fwi_root  = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/raw_historic_data/fwi"

output_dir = "/gws/ssde/j25b/bas_climate/users/clelland/model/full_bias_corr/inputs_raw"

# Your upstream sources seem to use -9999
DEFAULT_SRC_NODATA = -9999

# ------------------------------------------------
# BAND SELECTION
# ------------------------------------------------

BASELAND_BANDS = {
    1: "DEM",
    2: "slope",
    3: "aspect"
}

CMIP_BANDS = {
    1: "hurs",
    3: "pr",
    4: "rlds",
    5: "rsds",
    6: "sfcWind",
    7: "tas",
    8: "tasmax",
    9: "tasmin"
}

FWI_BANDS = {
    1: "BUI",
    2: "DC",
    3: "DMC",
    4: "FFMC",
    5: "FWI",
    11: "ISI"
}

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

def find_case_insensitive_folder(parent, target):
    for name in os.listdir(parent):
        if name.lower() == target.lower():
            return name
    return None


def align_to_reference(input_tif, ref_crs, ref_transform, ref_width, ref_height, ref_mask):
    """
    Reproject input_tif onto the reference grid and apply the reference mask so that
    only cells valid in the reference band 1 keep values; all others become NaN.

    The destination is initialized with NaNs and we pass both src_nodata and dst_nodata
    to reproject to ensure proper handling of holes / out-of-bounds regions.
    """
    with rasterio.open(input_tif) as src:
        src_nodata = src.nodata if src.nodata is not None else DEFAULT_SRC_NODATA

        # initialize with NaNs so non-overlapping areas remain NaN
        data = np.full((src.count, ref_height, ref_width), np.nan, dtype="float32")

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                src_nodata=src_nodata,     # <<<
                dst_nodata=np.nan,         # <<<
                resampling=Resampling.nearest
            )

        # enforce the reference mask: where ref is NaN, set all to NaN
        # ref_mask: True where valid, False where reference is NaN
        data[:, ~ref_mask] = np.nan     # <<<

        return data


def apply_ref_mask(arr, ref_mask):
    """
    Apply reference mask to array of shape (H, W) or (C, H, W).
    Keeps values only where ref_mask is True; elsewhere sets NaN.
    """
    if arr.ndim == 2:
        out = arr.astype("float32", copy=True)
        out[~ref_mask] = np.nan
        return out
    elif arr.ndim == 3:
        out = arr.astype("float32", copy=True)
        out[:, ~ref_mask] = np.nan
        return out
    else:
        raise ValueError("apply_ref_mask expects 2D or 3D array")

# ------------------------------------------------
# LOAD REFERENCE GRID + BUILD REFERENCE MASK (from band 1)
# ------------------------------------------------

with rasterio.open(baseland_file) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    ref_height    = ref.height
    ref_width     = ref.width

    # Build the mask from band 1 of the reference (DEM), which defines the "layout"
    ref_band1 = ref.read(1).astype("float32")
    # Convert known nodata to NaN
    ref_src_nodata = ref.nodata if ref.nodata is not None else DEFAULT_SRC_NODATA
    ref_band1[ref_band1 == ref_src_nodata] = np.nan
    # Mask: True where valid (not NaN), False where NaN
    ref_mask = ~np.isnan(ref_band1)  # <<<

# ------------------------------------------------
# PRELOAD BASELAND BANDS (clip to reference mask)
# ------------------------------------------------

with rasterio.open(baseland_file) as src:
    baseland_arrays = []
    baseland_names  = []

    src_nodata = src.nodata if src.nodata is not None else DEFAULT_SRC_NODATA

    for band, name in BASELAND_BANDS.items():
        arr = src.read(band).astype("float32")
        arr[arr == src_nodata] = np.nan
        arr = apply_ref_mask(arr, ref_mask)  # <<< ensure exact layout
        baseland_arrays.append(arr)
        baseland_names.append(name)

# ------------------------------------------------
# PRELOAD TEM (align + clip to reference mask)
# ------------------------------------------------

tem_stack = align_to_reference(
    tem_file,
    ref_crs,
    ref_transform,
    ref_width,
    ref_height,
    ref_mask,  # <<<
)

tem = tem_stack[0]  # if TEM is single-band; keep as-is if this is correct

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

os.makedirs(output_dir, exist_ok=True)

for root, model in zip(roots, models):
    cmip_model_folder = f"{root}"
    fwi_model_folder  = f"{root}"

    for scenario in scenarios:
        cmip_scenario = find_case_insensitive_folder(
            os.path.join(cmip_root, cmip_model_folder),
            scenario
        )

        fwi_scenario = find_case_insensitive_folder(
            os.path.join(fwi_root, fwi_model_folder),
            scenario
        )

        if cmip_scenario is None or fwi_scenario is None:
            print("Scenario folder missing:", scenario)
            continue

        for year in years:
            for month in months:
                cmip_file = os.path.join(
                    cmip_root,
                    cmip_model_folder,
                    cmip_scenario,
                    f"{model}_{scenario}_{year}_{month}_all_cog.tif"
                )

                fwi_file = os.path.join(
                    fwi_root,
                    fwi_model_folder,
                    fwi_scenario,
                    f"{model}_{scenario}_{year}_{month}_cog.tif"
                )

                if not os.path.exists(cmip_file) or not os.path.exists(fwi_file):
                    print("Missing:", year, month)
                    continue

                arrays = []
                names  = []

                # baseland bands (already masked)
                arrays.extend(baseland_arrays)
                names.extend(baseland_names)

                # TEM (already masked)
                arrays.append(tem)
                names.append("b1")

                # CMIP (align + mask)
                cmip_stack = align_to_reference(
                    cmip_file,
                    ref_crs,
                    ref_transform,
                    ref_width,
                    ref_height,
                    ref_mask,  # <<<
                )

                for band, name in CMIP_BANDS.items():
                    arrays.append(cmip_stack[band - 1])
                    names.append(name)

                # FWI (align + mask)
                fwi_stack = align_to_reference(
                    fwi_file,
                    ref_crs,
                    ref_transform,
                    ref_width,
                    ref_height,
                    ref_mask,  # <<<
                )

                for band, name in FWI_BANDS.items():
                    arrays.append(fwi_stack[band - 1])
                    names.append(name)

                stack = np.stack(arrays)

                out_folder = os.path.join(output_dir, model, scenario)
                os.makedirs(out_folder, exist_ok=True)

                out_file = os.path.join(
                    out_folder,
                    f"{model}_{scenario}_{year}_{month:02d}.tif"
                )

                meta = {
                    "driver": "GTiff",
                    "height": ref_height,
                    "width": ref_width,
                    "count": stack.shape[0],
                    "dtype": "float32",
                    "crs": ref_crs,
                    "transform": ref_transform,
                    "nodata": np.nan,  # works well with Float32 GeoTIFFs
                }

                with rasterio.open(out_file, "w", **meta) as dst:
                    for i in range(stack.shape[0]):
                        dst.write(stack[i], i + 1)
                        dst.set_band_description(i + 1, names[i])

                print("Saved:", out_file)