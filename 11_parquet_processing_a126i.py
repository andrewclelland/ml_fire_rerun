#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
# UPDATE: Pointing to the new FWI directory
OUT_DIR = "/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/ACCESS-CM2/ssp126"

PARQUET_DIR    = Path(OUT_DIR) / "parquet_coarse_grids_annual"
# NOTE: Saves tabular .parquet files. This is your main ML training dataset containing 
# the aggregated predictors (weather/terrain) and the final binary 'burned_label' for each cell.

COARSE_TIF_DIR = Path(OUT_DIR) / "tifs_coarse_grids_annual"
# NOTE: Saves 1-degree .tif raster maps showing ONLY the binary burned/unburned classification. 
# Useful for quickly viewing the 5% threshold results in GIS software.

COARSE_SHP_DIR = Path(OUT_DIR) / "shp_coarse_grids_annual"
# NOTE: Saves 1-degree vector polygon grids (.shp). Contains the cell ID and 'burned_label'. 
# Great for overlaying your classification grid on top of other maps in QGIS/ArcGIS.

os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(COARSE_TIF_DIR, exist_ok=True)
os.makedirs(COARSE_SHP_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
WANTED = [
    "DEM", "slope", "aspect", "b1", "relative_humidity",
    "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", #"drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code", "fire_weather_index",
    "initial_fire_spread_index",
]

GRID_SIZES_DEG      = [1]
BURNED_THRESHOLD    = 0.05
FRACTION_BAND_NAME  = "fraction"
WRITE_QA_LABEL_ON_4KM = False

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

WANTED_NORM   = [_norm(x) for x in WANTED]
FRACTION_NORM = _norm(FRACTION_BAND_NAME)

# FIXED: Updated regex to properly match the '_new_fwi_' string in the filename
name_re = re.compile(r"ACCESS-CM2_ssp126_(\d{4})_(\d{1,2}).tif$", re.IGNORECASE)

def parse_year_month(path: Path):
    m = name_re.search(path.name)
    return (int(m.group(1)), int(m.group(2))) if m else None

def map_band_indices_by_name(ds: rio.DatasetReader):
    mapping = {}
    descs = ds.descriptions
    for i, d in enumerate(descs, start=1):
        if d is None: d = f"B{i}"
        mapping[_norm(d)] = i
    return mapping, descs

def compute_lonlat_grid(ds: rio.DatasetReader):
    h, w = ds.height, ds.width
    rows, cols = np.indices((h, w))
    xs, ys = rio.transform.xy(ds.transform, rows, cols, offset="center")
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if ds.crs is None: raise RuntimeError("Dataset has no CRS")
    if ds.crs.to_epsg() == 4326: return x.astype(np.float32), y.astype(np.float32)
    transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon.astype(np.float32), lat.astype(np.float32)

def mode_ignore_nan(x: pd.Series):
    x = x.dropna()
    return x.value_counts().idxmax() if not x.empty else np.nan

def aggregate_to_coarse_grids_annual(
    year: int, ds: rio.DatasetReader, predictors_stack: np.ndarray,
    predictor_names: list, annual_frac: np.ndarray, lon: np.ndarray, lat: np.ndarray,
    grid_sizes_deg, burned_threshold, parquet_dir, coarse_tif_dir, coarse_shp_dir, base_name
):
    H, W = ds.height, ds.width
    N = H * W
    lon_flat, lat_flat, frac_flat = lon.ravel(), lat.ravel(), annual_frac.ravel()

    binary_4km_flat = np.zeros_like(frac_flat, dtype=np.uint8)
    #valid_frac = ~np.isnan(frac_flat) # OG
    valid_frac = np.isfinite(annual_frac).ravel()
    binary_4km_flat[valid_frac & (frac_flat > 0)] = 1 

    pred_flat = {name: band.ravel() for name, band in zip(predictor_names, predictors_stack)}
    valid_idx = np.nonzero(valid_frac)[0]
    if valid_idx.size == 0: return

    frac_valid, bin_valid = frac_flat[valid_frac], binary_4km_flat[valid_frac]
    lon_valid, lat_valid = lon_flat[valid_frac], lat_flat[valid_frac]
    pred_valid = {name: arr[valid_frac] for name, arr in pred_flat.items()}

    for size_deg in grid_sizes_deg:
        # Check if parquet exists before aggregating
        parquet_path = parquet_dir / f"{base_name}_grid{size_deg}deg.parquet"
        if parquet_path.exists():
            print(f"[SKIP] Parquet exists: {parquet_path}")
            continue

        big_lon = size_deg * np.floor(lon_valid / size_deg)
        big_lat = size_deg * np.floor(lat_valid / size_deg)

        df_dict = {
            "big_lon": big_lon.astype(np.float32), "big_lat": big_lat.astype(np.float32),
            "burned_4km": bin_valid, "frac_4km": frac_valid, "flat_idx": valid_idx,
        }
        for name in predictor_names: df_dict[name] = pred_valid[name].astype(np.float32)

        df = pd.DataFrame(df_dict)
        agg_dict = {"burned_4km": "mean", "frac_4km": "mean"}
        for name in predictor_names:
            if name == "b1": agg_dict[name] = mode_ignore_nan
            elif name in ["relative_humidity", "total_precipitation_sum"]: agg_dict[name] = "min"
            elif name in ["temperature_2m", "temperature_2m_min", "temperature_2m_max", 
                          #"build_up_index", "drought_code", "duff_moisture_code", 
                          "build_up_index", "duff_moisture_code", 
                          "fine_fuel_moisture_code", "fire_weather_index", "initial_fire_spread_index"]:
                agg_dict[name] = "max"
            else: agg_dict[name] = "mean"

        grouped = df.groupby(["big_lon", "big_lat"], as_index=False).agg(agg_dict)
        grouped = grouped.rename(columns={"burned_4km": "burned_frac_4km"})
        grouped["burned_label"] = (grouped["burned_frac_4km"] >= burned_threshold).astype(np.uint8)
        grouped = grouped.sort_values(["big_lat", "big_lon"]).reset_index(drop=True)
        grouped["ID"] = np.arange(len(grouped), dtype=np.int64)
        grouped["year"], grouped["grid_deg"] = year, size_deg

        grouped.to_parquet(parquet_path, index=False)
        print(f"[PARQUET] Saved {parquet_path}")

        # --- GeoTIFF Output ---
        tif_path = coarse_tif_dir / f"{base_name}_grid{size_deg}deg_epsg4326_burned_unburned.tif"
        if not tif_path.exists():
            min_lon, max_lon = grouped["big_lon"].min(), grouped["big_lon"].max() + size_deg
            min_lat, max_lat = grouped["big_lat"].min(), grouped["big_lat"].max() + size_deg
            transform = from_origin(min_lon, max_lat, size_deg, size_deg)
            width, height = int(np.ceil((max_lon - min_lon) / size_deg)), int(np.ceil((max_lat - min_lat) / size_deg))

            shapes = [(Polygon([(l, t), (l+size_deg, t), (l+size_deg, t+size_deg), (l, t+size_deg)]), int(v)) 
                      for l, t, v in zip(grouped["big_lon"], grouped["big_lat"], grouped["burned_label"])]
            
            coarse_raster = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=255, dtype="uint8")
            profile = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "uint8",
                       "crs": "EPSG:4326", "transform": transform, "nodata": 255, "compress": "LZW"}
            with rio.open(tif_path, "w", **profile) as dst: dst.write(coarse_raster, 1)
            print(f"[TIF] Saved {tif_path}")

        # --- Shapefile Output ---
        shp_path = coarse_shp_dir / f"{base_name}_grid{size_deg}deg_cells_epsg4326.shp"
        if not shp_path.exists():
            geoms = [Polygon([(l, t), (l+size_deg, t), (l+size_deg, t+size_deg), (l, t+size_deg)]) 
                     for l, t in zip(grouped["big_lon"], grouped["big_lat"])]
            shp_gdf = gpd.GeoDataFrame({"ID": grouped["ID"], "burned_label": grouped["burned_label"]}, 
                                        geometry=geoms, crs="EPSG:4326")
            shp_gdf.to_file(shp_path)
            print(f"[SHP] Saved {shp_path}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
# FIXED: Updated glob pattern to catch _new_fwi_ files
monthly_tifs = sorted(Path(OUT_DIR).glob("ACCESS-CM2_ssp126_*.tif"))
year_to_paths = defaultdict(list)
for p in monthly_tifs:
    ym = parse_year_month(p)
    if ym: year_to_paths[ym[0]].append((ym[1], p))

for year in sorted(year_to_paths.keys()):
    # FIXED: Updated the expected parquet name to match the new output pattern
    expected_parquet = PARQUET_DIR / f"ACCESS-CM2_ssp126_{year}_annual_grid1deg.parquet"
    if expected_parquet.exists():
        print(f"[SKIP YEAR] All outputs for {year} appear to exist.")
        continue

    month_paths = sorted(year_to_paths[year], key=lambda x: x[0])
    print(f"\n[YEAR] {year} — Processing {len(month_paths)} files")

    with rio.open(month_paths[0][1]) as ds_template:
        H, W = ds_template.height, ds_template.width
        band_map, _ = map_band_indices_by_name(ds_template)
        
        predictor_indices, predictor_names = [], []
        for want_norm, want_orig in zip(WANTED_NORM, WANTED):
            if want_norm in band_map:
                predictor_indices.append(band_map[want_norm]); predictor_names.append(want_orig)
            else:
                for k_norm, idx in band_map.items():
                    if want_norm in k_norm or k_norm in want_norm:
                        predictor_indices.append(idx); predictor_names.append(want_orig); break
        
        #frac_idx = band_map.get(FRACTION_NORM)
        frac_idx = band_map.get('temperature2m')
        #if frac_idx is None: continue

        frac_months = []
        pred_months = {name: [] for name in predictor_names}

        for _, path in month_paths:
            with rio.open(path) as ds_m:
                for name, idx in zip(predictor_names, predictor_indices):
                    pred_months[name].append(ds_m.read(idx).astype(np.float32))
                frac_months.append(ds_m.read(frac_idx).astype(np.float32))

        annual_frac = np.nanmax(np.stack(frac_months), axis=0)
        predictor_arrays = [np.nanmean(np.stack(pred_months[n]), axis=0).astype(np.float32) for n in predictor_names]
        lon, lat = compute_lonlat_grid(ds_template)

        # FIXED: Updated base_name to include _new_fwi
        aggregate_to_coarse_grids_annual(
            year, ds_template, np.stack(predictor_arrays), predictor_names, annual_frac, lon, lat,
            GRID_SIZES_DEG, BURNED_THRESHOLD, PARQUET_DIR, COARSE_TIF_DIR, COARSE_SHP_DIR, f"ACCESS-CM2_ssp126_{year}_annual"
        )

print("\n[DONE]")