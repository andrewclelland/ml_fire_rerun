#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
from rasterio.warp import transform as rio_transform
import geopandas as gpd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

# ================== CONFIG ==================
# Point to the new TIFF files that contain the replaced FWI variables
IN_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/MRI-ESM2-0/ssp126")

# UPDATED: Point to the new prediction-only shapefiles
PRED_SHP_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/testing_inputs_final_indiv/MRI-ESM2-0/ssp126/predictions_only_shapefiles_annual")

# UPDATED: Output directory for the new dataset (renamed to reflect it's masked by predictions)
OUT_DATASET_DIR = Path("/gws/ssde/j25b/bas_climate/users/clelland/model/parquets_indiv/parquet_MRI-ESM2-0_ssp126")
OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

REPROJECT_TO_EPSG4326 = True

# Years to process
YEAR_MIN = 2026
YEAR_MAX = 2100

# UPDATED: Mask shapefile criterion: keep only cells where Stage 1 PREDICTED burnable (1)
PRED_LABEL_VALUE = 1
PRED_LABEL_FIELD_OVERRIDE = None  # set if you know exact field name

# Fraction band description/name candidates (searched in ds.descriptions)
FRACTION_BAND_DESC_CANDIDATES = ["fraction", "frac", "burn_fraction"]

# Pixel label from fraction
PIXEL_BURN_THRESHOLD = 0.5  # burned if fraction > 0.5, unburned if fraction < 0.5

# ================== HELPERS ==================
def sanitize_names(names):
    """Make unique, safe column names (avoid duplicates)."""
    seen = {}
    out = []
    for n in names:
        if n is None or str(n).strip() == "":
            n = "band"
        n0 = re.sub(r"[^a-zA-Z0-9_]", "_", str(n).strip())
        n0 = re.sub(r"_+", "_", n0).strip("_")
        if n0 == "":
            n0 = "band"
        if n0 in seen:
            seen[n0] += 1
            n0 = f"{n0}_{seen[n0]}"
        else:
            seen[n0] = 1
        out.append(n0)
    return out

# Regex to match the new filenames
name_re = re.compile(r"MRI-ESM2-0_ssp126_(\d{4})_(\d{1,2})\.tif$", re.IGNORECASE)
# UPDATED: Shapefile regex to match the new prediction-only naming convention
shp_re = re.compile(r"MRI-ESM2-0_ssp126_(\d{4})_annual_grid1deg_predictions_only\.shp$", re.IGNORECASE)

def parse_year_month(fname: str):
    m = name_re.search(fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def append_chunk_to_dataset(df: pd.DataFrame, root: Path):
    if not df.columns.is_unique:
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {dups}")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["year", "month"],
        use_dictionary=False
    )

def find_fraction_band_index(ds: rio.DatasetReader) -> int:
    """
    Return 0-based band index for fraction band by inspecting ds.descriptions.
    """
    descs = list(ds.descriptions) if ds.descriptions else [None] * ds.count
    descs_safe = sanitize_names([d if d else f"B{i}" for i, d in enumerate(descs, start=1)])
    descs_safe_lower = [d.lower() for d in descs_safe]

    for cand in FRACTION_BAND_DESC_CANDIDATES:
        cand = cand.lower()
        for i, d in enumerate(descs_safe_lower):
            if cand == d or cand in d:
                return i

    raise RuntimeError(
        "Could not find fraction band by description. "
        f"Band descriptions (sanitized): {descs_safe}"
    )

def build_lonlat(ds: rio.DatasetReader, xs, ys):
    if (
        REPROJECT_TO_EPSG4326
        and ds.crs is not None
        and ds.crs.to_string().upper() not in ("EPSG:4326", "OGC:CRS84")
    ):
        lons, lats = rio_transform(ds.crs, "EPSG:4326", xs, ys)
        return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)
    return xs.astype(np.float64), ys.astype(np.float64)

def find_pred_label_field(gdf: gpd.GeoDataFrame) -> str:
    """
    UPDATED: Find the 'pred_label' field from the Stage 1 predictions.
    """
    if PRED_LABEL_FIELD_OVERRIDE:
        if PRED_LABEL_FIELD_OVERRIDE not in gdf.columns:
            raise RuntimeError(f"Override pred label field '{PRED_LABEL_FIELD_OVERRIDE}' not in: {list(gdf.columns)}")
        return PRED_LABEL_FIELD_OVERRIDE

    cols_lower = {c.lower(): c for c in gdf.columns}

    # Only look for prediction columns now
    candidates = ["pred_label", "pred", "prediction"]
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]

    raise RuntimeError(f"Could not find pred_label field. Columns: {list(gdf.columns)}")

def raster_mask_from_predlabel(ds: rio.DatasetReader, shp_path: Path) -> np.ndarray:
    """
    UPDATED: Rasterize polygons where pred_label==1 onto ds grid -> boolean mask (H,W).
    """
    gdf = gpd.read_file(shp_path)
    lab_col = find_pred_label_field(gdf)

    lab_vals = pd.to_numeric(gdf[lab_col], errors="coerce")
    gdf_keep = gdf.loc[lab_vals == PRED_LABEL_VALUE].copy()

    if gdf_keep.empty:
        return np.zeros((ds.height, ds.width), dtype=bool)

    if ds.crs is None:
        raise RuntimeError(f"Raster has no CRS; cannot rasterize: {shp_path}")
    if gdf_keep.crs is None:
        raise RuntimeError(f"Shapefile has no CRS; cannot rasterize: {shp_path}")

    if gdf_keep.crs != ds.crs:
        gdf_keep = gdf_keep.to_crs(ds.crs)

    shapes = [(geom, 1) for geom in gdf_keep.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((ds.height, ds.width), dtype=bool)

    mask_u8 = rasterize(
        shapes=shapes,
        out_shape=(ds.height, ds.width),
        transform=ds.transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask_u8.astype(bool)

# ================== MAIN ==================
def main():
    tifs = sorted(IN_DIR.glob("MRI-ESM2-0_ssp126_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No monthly tif found in {IN_DIR}")

    # Filter to years 2001-2022
    todo = []
    for tif in tifs:
        y, m = parse_year_month(tif.name)
        if y is None:
            continue
        if y < YEAR_MIN or y > YEAR_MAX:
            continue
        todo.append((y, m, tif))
    todo.sort()

    if not todo:
        raise RuntimeError(f"No TIFFs found in year range {YEAR_MIN}-{YEAR_MAX}")

    # Cache the rasterized prediction mask per year
    year_mask_cache = {}

    canonical_cols = None

    # Global ratio counters
    burned_total = 0
    unburned_total = 0
    valid_lab_total = 0

    print(f"Scanning for prediction shapefiles in: {PRED_SHP_DIR}")

    for year, month, tif in tqdm(todo, desc="Building partitioned Parquet dataset (Stage 1 Pred mask)"):
        
        # ------------------------------------------------------------------
        # Check if output partition exists to avoid reprocessing
        # ------------------------------------------------------------------
        partition_path = OUT_DATASET_DIR / f"year={year}" / f"month={month}"
        if partition_path.exists() and any(partition_path.glob("*.parquet")):
            # print(f"[SKIP] {year}-{month} exists.") 
            continue

        # ------------------------------------------------------------------
        # Prepare inputs
        # ------------------------------------------------------------------
        # UPDATED: Name to match the predictions-only script
        shp_name = f"MRI-ESM2-0_ssp126_{year}_annual_grid1deg_predictions_only.shp"
        shp_path = PRED_SHP_DIR / shp_name
        
        if not shp_path.exists():
            print(f"\n[SKIP] {tif.name} (missing annual prediction shapefile: {shp_path})")
            continue

        with rio.open(tif) as ds:
            # band names
            band_names = list(ds.descriptions) if ds.descriptions else []
            if not any(band_names):
                band_names = [f"B{i}" for i in range(1, ds.count + 1)]
            safe_names = sanitize_names(band_names)

            # Predicted burnable mask per year (rasterized once)
            if year not in year_mask_cache:
                mask = raster_mask_from_predlabel(ds, shp_path)
                year_mask_cache[year] = mask
                print(f"\n[YEAR {year}] pred_label mask keeps {mask.sum():,} / {mask.size:,} pixels ({100*mask.mean():.2f}%)")
            else:
                mask = year_mask_cache[year]
                if mask.shape != (ds.height, ds.width):
                    raise RuntimeError(f"Mask shape mismatch for {year}: mask {mask.shape} vs raster {(ds.height, ds.width)}")

            if mask.sum() == 0:
                continue

            # Read raster (bands, H, W)
            data = ds.read().astype(np.float32)
            bands, h, w = data.shape

            # Flatten to (pixels, bands)
            arr2d = data.reshape(bands, -1).T

            # Keep only pixels with build_up_index not NaN (domain mask)
            build_col = None
            for s in safe_names:
                if "build" in s.lower() and "index" in s.lower():
                    build_col = s
                    break
            if build_col is None:
                raise ValueError(f"Could not find build_up_index band in: {tif.name}")

            build_idx = safe_names.index(build_col)
            build_vals = arr2d[:, build_idx]

            keep_mask = mask.reshape(-1) & (~np.isnan(build_vals))
            if not keep_mask.any():
                continue

            # Subset pixels
            arr_keep = arr2d[keep_mask, :]
            df = pd.DataFrame(arr_keep, columns=safe_names)

            # Coordinates for kept pixels
            rows = np.arange(h)
            cols = np.arange(w)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            xs, ys = rio.transform.xy(ds.transform, rr, cc, offset="center")
            xs = np.asarray(xs, dtype=np.float64).reshape(-1)[keep_mask]
            ys = np.asarray(ys, dtype=np.float64).reshape(-1)[keep_mask]
            lons, lats = build_lonlat(ds, xs, ys)

            df["longitude"] = lons
            df["latitude"] = lats
            df["year"] = year
            df["month"] = month

            # Canonical schema
            if canonical_cols is None:
                canonical_cols = list(safe_names)
                for extra in ["burned_pixel", "longitude", "latitude", "year", "month"]:
                    if extra not in canonical_cols:
                        canonical_cols.append(extra)
                if len(canonical_cols) != len(set(canonical_cols)):
                    raise RuntimeError(f"Canonical cols not unique: {canonical_cols}")

            for col in canonical_cols:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[canonical_cols]
            append_chunk_to_dataset(df, OUT_DATASET_DIR)

    print(f"\n✅ Done. Parquet dataset at:\n{OUT_DATASET_DIR}\n(partitioned by year=/month=)")

    # Global ratios
    print("\n=== Burned/Unburned pixel counts (filtered to predicted burnable 1° cells) ===")
    print(f"Valid labeled pixels (fraction != NaN and != {PIXEL_BURN_THRESHOLD}): {valid_lab_total:,}")
    print(f"Burned pixels    (fraction > {PIXEL_BURN_THRESHOLD}): {burned_total:,}")
    print(f"Unburned pixels  (fraction < {PIXEL_BURN_THRESHOLD}): {unburned_total:,}")

    if burned_total > 0:
        ratio = unburned_total / burned_total
        print(f"Unburned:Burned ratio = {ratio:.3f} : 1")
    else:
        print("Unburned:Burned ratio = inf (no burned pixels found)")

if __name__ == "__main__":
    main()