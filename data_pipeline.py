"""
Ship Routing Data Pipeline
==========================
Step 1: Load & clean AIS data (Danish Maritime Authority format)
Step 2: Fetch weather in BATCHES from Open-Meteo (free, fast, no API key)
Step 3: Fetch ocean currents from Copernicus Marine (optional)
Step 4: Generate fuel consumption labels using Admiralty Formula
Step 5: Save final training-ready dataset

WHY Open-Meteo instead of NOAA ERDDAP?
  - ERDDAP makes one HTTP request per row → 35 hours for 5k rows
  - Open-Meteo accepts batches of up to 1000 locations → done in seconds
  - Completely free, no account needed

Requirements:
    pip install pandas numpy requests tqdm

Optional (for Copernicus currents):
    pip install copernicusmarine
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

AIS_FILE    = "aisdk-2025-10-29.csv"   # your AIS file
OUTPUT_FILE = "ship_training_data.csv"

# Generic ship defaults
SHIP_DISPLACEMENT_TONNES = 50_000
ADMIRALTY_COEFFICIENT    = 600
SFOC_G_PER_KWH           = 180

# Sampling
SAMPLE_EVERY_N_ROWS = 10     # use every Nth AIS ping
MAX_ROWS            = 5000   # total rows after sampling (set None for full run)

# Open-Meteo endpoints (free, no key)
MARINE_API  = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE  = 500   # rows per API call (stay under the 1000-location limit)


# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN AIS DATA
# ─────────────────────────────────────────────

def load_ais(filepath: str) -> pd.DataFrame:
    """
    Load DMA AIS CSV in chunks to avoid out-of-memory errors.
    Download from: https://www.dma.dk/safety-at-sea/navigational-information/ais-data
    """
    print(f"[1/5] Loading AIS data from {filepath}...")

    CHUNK_SIZE = 100_000
    chunks = []
    total = 0
    target = (MAX_ROWS or 50_000) * SAMPLE_EVERY_N_ROWS * 4  # over-fetch before filtering

    for chunk in pd.read_csv(filepath, low_memory=False, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        total += len(chunk)
        if total >= target:
            break

    df = pd.concat(chunks, ignore_index=True)

    # ── Standardise column names ──
    col_map = {
        "# Timestamp": "timestamp", "Timestamp": "timestamp",
        "MMSI": "mmsi",
        "Latitude": "lat", "Longitude": "lon",
        "SOG": "sog", "COG": "cog", "Heading": "heading",
        "Draught": "draught", "Ship type": "ship_type",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Parse & filter ──
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df.dropna(subset=["timestamp", "lat", "lon", "sog"], inplace=True)
    df = df[
        df["lat"].between(-90, 90) &
        df["lon"].between(-180, 180) &
        df["sog"].between(0.5, 30)        # moving vessels only
    ]

    df = df.iloc[::SAMPLE_EVERY_N_ROWS].reset_index(drop=True)
    if MAX_ROWS:
        df = df.head(MAX_ROWS)

    keep = [c for c in ["timestamp","mmsi","lat","lon","sog","cog","heading","draught"] if c in df.columns]
    df = df[keep].copy()

    print(f"    -> {len(df):,} rows after cleaning")
    return df


# ─────────────────────────────────────────────
# STEP 2: FETCH WEATHER IN BATCHES (Open-Meteo)
# ─────────────────────────────────────────────
#
# Strategy:
#   Group AIS rows by (date, lat_bucket, lon_bucket).
#   For each unique location+date, make ONE marine API call that
#   returns a full 24-hour hourly timeseries.
#   Then join results back to AIS rows by nearest hour.
#
#   This reduces thousands of API calls to a few hundred at most.

def _round_coord(x, res=1.0):
    return round(x / res) * res


def fetch_weather_for_location(lat: float, lon: float, date_str: str) -> pd.DataFrame:
    """
    Fetch hourly wave + wind data for a single (lat, lon, date).
    Two API calls: marine (waves) + archive (wind ERA5).
    Returns a DataFrame indexed by hour.
    """
    wave_df = wind_df = None

    # ── Waves ──
    try:
        r = requests.get(MARINE_API, params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "wave_height,wave_period",
            "timezone": "UTC",
        }, timeout=20)
        if r.status_code == 200:
            d = r.json()
            wave_df = pd.DataFrame({
                "hour":        pd.to_datetime(d["hourly"]["time"]),
                "wave_height": d["hourly"]["wave_height"],
                "wave_period": d["hourly"]["wave_period"],
            })
    except Exception:
        pass

    # ── Wind (ERA5 reanalysis — works for any past date) ──
    try:
        r2 = requests.get(ARCHIVE_API, params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "windspeed_10m,winddirection_10m",
            "timezone": "UTC",
            "windspeed_unit": "ms",
        }, timeout=20)
        if r2.status_code == 200:
            d2 = r2.json()
            wind_df = pd.DataFrame({
                "hour":       pd.to_datetime(d2["hourly"]["time"]),
                "wind_speed": d2["hourly"]["windspeed_10m"],
                "wind_dir":   d2["hourly"]["winddirection_10m"],
            })
    except Exception:
        pass

    # ── Merge both ──
    if wave_df is not None and wind_df is not None:
        return pd.merge(wave_df, wind_df, on="hour", how="outer")
    elif wave_df is not None:
        wave_df["wind_speed"] = np.nan
        wave_df["wind_dir"]   = np.nan
        return wave_df
    elif wind_df is not None:
        wind_df["wave_height"] = np.nan
        wind_df["wave_period"] = np.nan
        return wind_df
    else:
        return pd.DataFrame()


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach weather to each AIS row using batch grid downloads.
    Groups rows by (1-degree grid cell, date) to minimise API calls.
    """
    print("[2/5] Fetching weather data via Open-Meteo (batch mode)...")

    GRID = 1.0
    df = df.copy()
    df["_lat_g"] = df["lat"].apply(lambda x: _round_coord(x, GRID))
    df["_lon_g"] = df["lon"].apply(lambda x: _round_coord(x, GRID))
    df["_date"]  = df["timestamp"].dt.strftime("%Y-%m-%d")

    unique_locs = df[["_lat_g","_lon_g","_date"]].drop_duplicates().values.tolist()
    print(f"    -> {len(unique_locs)} unique grid cells to fetch (instead of {len(df):,} individual calls)")

    # ── Fetch each unique grid cell ──
    cache = {}
    for lat_g, lon_g, date_str in tqdm(unique_locs, desc="  Fetching grids"):
        key = (lat_g, lon_g, date_str)
        cache[key] = fetch_weather_for_location(lat_g, lon_g, date_str)
        time.sleep(0.05)  # gentle rate limiting

    # ── Join back to AIS rows by nearest hour ──
    wave_heights, wave_periods, wind_speeds, wind_dirs = [], [], [], []

    for _, row in df.iterrows():
        key = (row["_lat_g"], row["_lon_g"], row["_date"])
        wdf = cache.get(key)

        if wdf is not None and not wdf.empty and "hour" in wdf.columns:
            ts = row["timestamp"].replace(minute=0, second=0, microsecond=0)
            match = wdf[wdf["hour"] == ts]
            if match.empty:
                wdf2 = wdf.copy()
                wdf2["_diff"] = (wdf2["hour"] - row["timestamp"]).abs()
                match = wdf2.nsmallest(1, "_diff")

            wave_heights.append(match["wave_height"].values[0] if "wave_height" in match.columns else np.nan)
            wave_periods.append(match["wave_period"].values[0]  if "wave_period" in match.columns else np.nan)
            wind_speeds.append(match["wind_speed"].values[0]    if "wind_speed" in match.columns else np.nan)
            wind_dirs.append(match["wind_dir"].values[0]        if "wind_dir" in match.columns else np.nan)
        else:
            wave_heights.append(np.nan)
            wave_periods.append(np.nan)
            wind_speeds.append(np.nan)
            wind_dirs.append(np.nan)

    df["wave_height_m"] = wave_heights
    df["wave_period_s"] = wave_periods
    df["wind_speed_ms"] = wind_speeds
    df["wind_dir_deg"]  = wind_dirs
    df.drop(columns=["_lat_g","_lon_g","_date"], inplace=True)

    filled = df["wave_height_m"].notna().sum()
    print(f"    -> Weather joined: {filled:,}/{len(df):,} rows have data")
    return df


# ─────────────────────────────────────────────
# STEP 3: OCEAN CURRENTS (Copernicus — optional)
# ─────────────────────────────────────────────

def add_currents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ocean surface currents.
    Uses copernicusmarine if installed, otherwise defaults to zero.
    Zero currents still produce a good fuel model — this is optional.
    """
    print("[3/5] Ocean currents...")

    try:
        import copernicusmarine as cm

        date_str = df["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
        lat_min, lat_max = df["lat"].min() - 1, df["lat"].max() + 1
        lon_min, lon_max = df["lon"].min() - 1, df["lon"].max() + 1

        print(f"    Downloading CMEMS current grid for {date_str}...")
        ds = cm.open_dataset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["uo", "vo"],
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T23:59:59",
            minimum_depth=0, maximum_depth=1,
        )

        def lookup_current(lat, lon):
            try:
                pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
                return float(pt["uo"].values.mean()), float(pt["vo"].values.mean())
            except Exception:
                return 0.0, 0.0

        results = [lookup_current(r["lat"], r["lon"]) for _, r in tqdm(df.iterrows(), desc="  Joining currents", total=len(df))]
        df["current_u_ms"] = [r[0] for r in results]
        df["current_v_ms"] = [r[1] for r in results]
        print("    -> Copernicus currents added")

    except ImportError:
        print("    [info] copernicusmarine not installed -> using zero currents (fine for now)")
        df["current_u_ms"] = 0.0
        df["current_v_ms"] = 0.0
    except Exception as e:
        print(f"    [warn] Currents failed ({e}) -> using zeros")
        df["current_u_ms"] = 0.0
        df["current_v_ms"] = 0.0

    return df


# ─────────────────────────────────────────────
# STEP 4: FEATURES + FUEL LABEL
# ─────────────────────────────────────────────

def compute_relative_wind_angle(heading: float, wind_dir: float) -> float:
    """0 = headwind, 90 = beam wind, 180 = tailwind."""
    angle = (wind_dir - heading + 360) % 360
    return 360 - angle if angle > 180 else angle


def speed_through_water(sog_knots, current_u, current_v, heading_deg) -> float:
    """Remove current vector from SOG to get Speed Through Water."""
    MS_TO_KN = 1.94384
    hdg = np.radians(heading_deg)
    sog_ms = sog_knots / MS_TO_KN
    stw_u = sog_ms * np.sin(hdg) - current_u
    stw_v = sog_ms * np.cos(hdg) - current_v
    return np.sqrt(stw_u**2 + stw_v**2) * MS_TO_KN


def estimate_fuel_burn(stw_knots, wave_height_m=0, wind_speed_ms=0, rel_wind_deg=90) -> dict:
    """Admiralty Formula + wave/wind resistance corrections."""
    if stw_knots <= 0 or np.isnan(stw_knots):
        return {"power_kw": 0.0, "fuel_tph": 0.0}

    calm_power = (SHIP_DISPLACEMENT_TONNES**(2/3) * stw_knots**3) / ADMIRALTY_COEFFICIENT

    wh = 0 if (wave_height_m is None or np.isnan(wave_height_m)) else wave_height_m
    ws = 0 if (wind_speed_ms  is None or np.isnan(wind_speed_ms))  else wind_speed_ms
    ra = np.radians(90 if (rel_wind_deg is None or np.isnan(rel_wind_deg)) else rel_wind_deg)

    wave_factor = 1 + 0.05 * wh**2
    wind_factor = max(1 + 0.002 * ws**2 * np.cos(np.pi - ra), 0.95)

    power = calm_power * wave_factor * wind_factor
    fuel  = (power * SFOC_G_PER_KWH) / 1_000_000

    return {"power_kw": round(power, 2), "fuel_tph": round(fuel, 4)}


def add_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Computing features and fuel labels...")

    hdg_col = "heading" if "heading" in df.columns else "cog"

    rel_winds, stws, powers, fuels = [], [], [], []

    for _, r in df.iterrows():
        hdg = r.get(hdg_col, r.get("cog", 0))
        if pd.isna(hdg):
            hdg = r.get("cog", 0)
        if pd.isna(hdg):
            hdg = 0

        rw  = compute_relative_wind_angle(hdg, r.get("wind_dir_deg", 90) or 90)
        stw = speed_through_water(r["sog"], r.get("current_u_ms", 0) or 0,
                                  r.get("current_v_ms", 0) or 0, hdg)
        fb  = estimate_fuel_burn(stw,
                                 r.get("wave_height_m") or 0,
                                 r.get("wind_speed_ms") or 0,
                                 rw)

        rel_winds.append(round(rw, 1))
        stws.append(round(stw, 2))
        powers.append(fb["power_kw"])
        fuels.append(fb["fuel_tph"])

    df = df.copy()
    df["rel_wind_angle_deg"] = rel_winds
    df["stw_knots"]          = stws
    df["power_kw"]           = powers
    df["fuel_tph"]           = fuels   # <- TARGET VARIABLE (Y)

    return df


# ─────────────────────────────────────────────
# STEP 5: SAVE
# ─────────────────────────────────────────────

FINAL_COLS = [
    "sog", "stw_knots", "heading", "cog",
    "wave_height_m", "wave_period_s",
    "wind_speed_ms", "wind_dir_deg", "rel_wind_angle_deg",
    "current_u_ms", "current_v_ms", "draught",
    "fuel_tph",                        # target
    "timestamp", "mmsi", "lat", "lon", "power_kw",
]

def save_dataset(df: pd.DataFrame, path: str):
    print("[5/5] Saving dataset...")
    cols = [c for c in FINAL_COLS if c in df.columns]
    out  = df[cols].dropna(subset=["sog", "fuel_tph"])
    out.to_csv(path, index=False)
    print(f"    -> {len(out):,} rows saved to {path}")
    print("\n" + out.describe().round(3).to_string())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline(ais_file=AIS_FILE, output_file=OUTPUT_FILE):
    df = load_ais(ais_file)
    df = add_weather(df)
    df = add_currents(df)
    df = add_features_and_labels(df)
    save_dataset(df, output_file)
    print("\nPipeline complete! Run 02_train_model.py next.")
    return df


if __name__ == "__main__":
    run_pipeline()