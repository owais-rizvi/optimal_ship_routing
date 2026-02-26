"""
Ship Routing Algorithm  (with land mask)
========================================
Uses the trained XGBoost fuel model + A* search to find minimum-fuel
ocean route between two ports. Land cells are blocked automatically.

Requirements:
    pip install pandas numpy joblib requests tqdm shapely
"""

import numpy as np
import pandas as pd
import joblib
import requests
import heapq
import time
import json
from datetime import datetime, timedelta
from shapely.geometry import Point, shape
from shapely.ops import unary_union

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_FILE       = "fuel_model.joblib"
SHIP_SPEED_KNOTS = 12.0
GRID_RES         = 2.0      # degrees (2-3 for ocean routes, 0.5-1 for coastal)

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
MARINE_API  = "https://marine-api.open-meteo.com/v1/marine"

# Natural Earth low-res land polygons (tiny download, ~500KB, no auth needed)
LAND_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"
LAND_CACHE_FILE  = "ne_land.geojson"


# ─────────────────────────────────────────────
# LAND MASK
# ─────────────────────────────────────────────

def load_land_mask():
    """
    Download (once) and cache Natural Earth land polygons.
    Returns a single merged Shapely geometry for fast point-in-polygon tests.
    """
    if not __import__("os").path.exists(LAND_CACHE_FILE):
        print("  Downloading land mask (one-time, ~500KB)...")
        r = requests.get(LAND_GEOJSON_URL, timeout=30)
        r.raise_for_status()
        with open(LAND_CACHE_FILE, "w") as f:
            f.write(r.text)
        print(f"  Land mask saved to {LAND_CACHE_FILE}")

    with open(LAND_CACHE_FILE) as f:
        gj = json.load(f)

    polygons = [shape(feat["geometry"]) for feat in gj["features"]]
    land = unary_union(polygons)
    print(f"  Land mask loaded ({len(polygons)} polygons)")
    return land


def build_ocean_mask(grid, land):
    """
    Pre-compute a boolean array: True = ocean (navigable), False = land.
    Much faster than per-step point-in-polygon during search.
    """
    print("  Building ocean grid mask...", end=" ", flush=True)
    mask = np.ones((grid.n_rows, grid.n_cols), dtype=bool)
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            lat, lon = grid.latlon(r, c)
            if land.contains(Point(lon, lat)):   # Shapely uses (x=lon, y=lat)
                mask[r, c] = False
    ocean_cells = mask.sum()
    print(f"done. {ocean_cells:,}/{grid.n_rows*grid.n_cols:,} cells are ocean.")
    return mask


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(path):
    saved = joblib.load(path)
    return saved["model"], saved["features"]


# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl)*np.cos(p2)
    y = np.cos(p1)*np.sin(p2) - np.sin(p1)*np.cos(p2)*np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def rel_wind(heading, wind_dir):
    a = (wind_dir - heading + 360) % 360
    return 360 - a if a > 180 else a


# ─────────────────────────────────────────────
# INTEGER GRID
# ─────────────────────────────────────────────

class Grid:
    def __init__(self, lat_min, lat_max, lon_min, lon_max, res):
        self.res     = res
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.n_rows  = int(round((lat_max - lat_min) / res)) + 1
        self.n_cols  = int(round((lon_max - lon_min) / res)) + 1

    def latlon(self, r, c):
        return (
            round(self.lat_min + r * self.res, 6),
            round(self.lon_min + c * self.res, 6),
        )

    def rowcol(self, lat, lon):
        r = int(round((lat - self.lat_min) / self.res))
        c = int(round((lon - self.lon_min) / self.res))
        return (max(0, min(r, self.n_rows-1)),
                max(0, min(c, self.n_cols-1)))

    def valid(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def neighbors(self, r, c, ocean_mask):
        """8-connected ocean neighbors only."""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.valid(nr, nc) and ocean_mask[nr, nc]:
                    yield nr, nc

    def size(self):
        return self.n_rows * self.n_cols


# ─────────────────────────────────────────────
# FUEL PREDICTION
# ─────────────────────────────────────────────

def predict_fuel_rate(model, features, speed, heading, weather):
    rw = rel_wind(heading, weather.get("wind_dir_deg", 0))
    row = {
        "sog":                speed,
        "stw_knots":          speed,
        "wave_height_m":      weather.get("wave_height_m", 1.0),
        "wave_period_s":      weather.get("wave_period_s", 8.0),
        "wind_speed_ms":      weather.get("wind_speed_ms", 5.0),
        "rel_wind_angle_deg": rw,
        "current_u_ms":       0.0,
        "current_v_ms":       0.0,
    }
    X = np.array([row.get(f, 0) for f in features]).reshape(1, -1)
    return float(model.predict(X)[0])


# ─────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────

NEUTRAL_WEATHER = {"wave_height_m": 1.0, "wave_period_s": 8.0,
                   "wind_speed_ms": 5.0, "wind_dir_deg": 0.0}

def fetch_weather(lat, lon, dt):
    date_str = dt.strftime("%Y-%m-%d")
    hour = dt.replace(minute=0, second=0, microsecond=0)
    wh = wp = ws = wd = None
    try:
        r = requests.get(MARINE_API, params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "wave_height,wave_period", "timezone": "UTC",
        }, timeout=15)
        if r.status_code == 200:
            d = r.json()
            hrs = pd.to_datetime(d["hourly"]["time"])
            i = (hrs - hour).abs().argmin()
            wh = d["hourly"]["wave_height"][i]
            wp = d["hourly"]["wave_period"][i]
    except Exception:
        pass
    try:
        r2 = requests.get(ARCHIVE_API, params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "windspeed_10m,winddirection_10m",
            "timezone": "UTC", "windspeed_unit": "ms",
        }, timeout=15)
        if r2.status_code == 200:
            d2 = r2.json()
            hrs = pd.to_datetime(d2["hourly"]["time"])
            i = (hrs - hour).abs().argmin()
            ws = d2["hourly"]["windspeed_10m"][i]
            wd = d2["hourly"]["winddirection_10m"][i]
    except Exception:
        pass
    return {
        "wave_height_m": wh if wh is not None else 1.0,
        "wave_period_s": wp if wp is not None else 8.0,
        "wind_speed_ms": ws if ws is not None else 5.0,
        "wind_dir_deg":  wd if wd is not None else 0.0,
    }


# ─────────────────────────────────────────────
# A* ROUTER
# ─────────────────────────────────────────────

def astar_route(start_latlon, end_latlon, model, features,
                departure_time, land_geometry,
                speed=SHIP_SPEED_KNOTS, grid_res=GRID_RES,
                use_weather=False):

    s_lat, s_lon = start_latlon
    e_lat, e_lon = end_latlon

    pad = grid_res * 4
    lat_min = min(s_lat, e_lat) - pad
    lat_max = max(s_lat, e_lat) + pad
    lon_min = min(s_lon, e_lon) - pad
    lon_max = max(s_lon, e_lon) + pad

    grid = Grid(lat_min, lat_max, lon_min, lon_max, grid_res)
    print(f"\nRouting {start_latlon} -> {end_latlon}")
    print(f"Grid: {grid.n_rows}x{grid.n_cols} = {grid.size()} nodes | res={grid_res}°")

    # Build land mask
    ocean_mask = build_ocean_mask(grid, land_geometry)

    sr, sc = grid.rowcol(s_lat, s_lon)
    er, ec = grid.rowcol(e_lat, e_lon)

    # If start/end are on land, nudge to nearest ocean cell
    def nearest_ocean(r, c):
        if ocean_mask[r, c]:
            return r, c
        for radius in range(1, 10):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    nr, nc = r+dr, c+dc
                    if grid.valid(nr, nc) and ocean_mask[nr, nc]:
                        return nr, nc
        return r, c

    sr, sc = nearest_ocean(sr, sc)
    er, ec = nearest_ocean(er, ec)

    print(f"Start: {grid.latlon(sr,sc)} | End: {grid.latlon(er,ec)}")

    weather_cache = {}

    def get_weather(r, c, dt):
        key = (r, c, dt.strftime("%Y-%m-%dT%H"))
        if key not in weather_cache:
            if use_weather:
                la, lo = grid.latlon(r, c)
                weather_cache[key] = fetch_weather(la, lo, dt)
                time.sleep(0.05)
            else:
                weather_cache[key] = NEUTRAL_WEATHER
        return weather_cache[key]

    def heuristic(r, c):
        la, lo = grid.latlon(r, c)
        ea, eo = grid.latlon(er, ec)
        dist = haversine_nm(la, lo, ea, eo)
        hrs = dist / speed
        hdg = bearing_deg(la, lo, ea, eo)
        calm = {"wave_height_m":0,"wave_period_s":8,"wind_speed_ms":0,"wind_dir_deg":0}
        rate = predict_fuel_rate(model, features, speed, hdg, calm)
        return rate * hrs

    heap = [(heuristic(sr,sc), 0.0, sr, sc, [(sr,sc)], departure_time)]
    best_g = {}
    iters = 0
    MAX_ITERS = 100_000

    print(f"Searching (max {MAX_ITERS:,} iters, ocean cells only)...")

    while heap and iters < MAX_ITERS:
        f, g, r, c, path, t = heapq.heappop(heap)
        iters += 1

        if best_g.get((r,c), float("inf")) <= g:
            continue
        best_g[(r,c)] = g

        if iters % 3000 == 0:
            la, lo = grid.latlon(r, c)
            ea, eo = grid.latlon(er, ec)
            print(f"  iter {iters:,} | fuel={g:.3f}t | {haversine_nm(la,lo,ea,eo):.0f} nm left")

        # Goal check
        if r == er and c == ec:
            route_ll = [grid.latlon(pr,pc) for pr,pc in path]
            total_dist = sum(haversine_nm(*route_ll[i], *route_ll[i+1])
                             for i in range(len(route_ll)-1))
            travel_hrs = (t - departure_time).total_seconds() / 3600
            print(f"\n{'='*52}")
            print(f"  ROUTE FOUND  ({iters:,} iterations)")
            print(f"  Waypoints:   {len(route_ll)}")
            print(f"  Distance:    {total_dist:.1f} nm")
            print(f"  Travel time: {travel_hrs:.1f} hrs ({travel_hrs/24:.1f} days)")
            print(f"  Total fuel:  {g:.3f} tonnes")
            print(f"  ETA:         {t.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"{'='*52}")
            return {
                "route": route_ll,
                "total_fuel_tonnes": round(g, 3),
                "total_distance_nm": round(total_dist, 1),
                "eta": t,
            }

        for nr, nc in grid.neighbors(r, c, ocean_mask):
            if best_g.get((nr,nc), float("inf")) <= g:
                continue
            la,  lo  = grid.latlon(r,  c)
            nla, nlo = grid.latlon(nr, nc)
            dist = haversine_nm(la, lo, nla, nlo)
            hrs  = dist / speed
            hdg  = bearing_deg(la, lo, nla, nlo)
            nt   = t + timedelta(hours=hrs)
            w    = get_weather(nr, nc, nt)
            rate = predict_fuel_rate(model, features, speed, hdg, w)
            ng   = g + rate * hrs
            nf   = ng + heuristic(nr, nc)
            heapq.heappush(heap, (nf, ng, nr, nc, path+[(nr,nc)], nt))

    print(f"  No route found after {iters:,} iters.")
    return {}


# ─────────────────────────────────────────────
# GREAT CIRCLE BASELINE
# ─────────────────────────────────────────────

def great_circle_fuel(start, end, model, features, departure_time,
                      speed=SHIP_SPEED_KNOTS, n=20):
    lats = np.linspace(start[0], end[0], n+1)
    lons = np.linspace(start[1], end[1], n+1)
    total = 0.0
    t = departure_time
    for i in range(n):
        dist = haversine_nm(lats[i], lons[i], lats[i+1], lons[i+1])
        hdg  = bearing_deg(lats[i], lons[i], lats[i+1], lons[i+1])
        hrs  = dist / speed
        rate = predict_fuel_rate(model, features, speed, hdg, NEUTRAL_WEATHER)
        total += rate * hrs
        t += timedelta(hours=hrs)
    return {
        "route": list(zip(lats.round(4), lons.round(4))),
        "total_fuel_tonnes": round(total, 3),
        "total_distance_nm": round(haversine_nm(*start, *end), 1),
        "eta": t,
    }


# ─────────────────────────────────────────────
# SAVE ROUTE
# ─────────────────────────────────────────────

def save_route_csv(result, filename):
    if not result:
        return
    rows = [{"lat": p[0], "lon": p[1], "waypoint": i+1}
            for i, p in enumerate(result["route"])]
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"Saved -> {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Install shapely if missing
    try:
        from shapely.geometry import Point
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])

    model, features = load_model(MODEL_FILE)
    print(f"Model loaded | features: {features}")

    # Load land mask (downloads once, cached as ne_land.geojson)
    print("\nLoading land mask...")
    land = load_land_mask()

    ROTTERDAM = (51.9,  4.5)
    NEW_YORK  = (40.7, -74.0)
    DEPARTURE = datetime(2024, 6, 15, 0, 0)

    # ── Baseline ──
    print("\n--- BASELINE: Great Circle ---")
    gc = great_circle_fuel(ROTTERDAM, NEW_YORK, model, features, DEPARTURE)
    print(f"  Distance: {gc['total_distance_nm']} nm | Fuel: {gc['total_fuel_tonnes']} t")
    save_route_csv(gc, "great_circle_route.csv")

    # ── Optimised (ocean only) ──
    # use_weather=False: instant test with neutral conditions
    # use_weather=True:  real weather from Open-Meteo (takes a few minutes)
    result = astar_route(
        start_latlon   = ROTTERDAM,
        end_latlon     = NEW_YORK,
        model          = model,
        features       = features,
        departure_time = DEPARTURE,
        land_geometry  = land,
        speed          = SHIP_SPEED_KNOTS,
        grid_res       = 2.0,
        use_weather    = False,
    )

    if result:
        saving = gc["total_fuel_tonnes"] - result["total_fuel_tonnes"]
        pct    = saving / gc["total_fuel_tonnes"] * 100
        print(f"\n  Fuel saving vs great circle: {saving:.2f} t ({pct:.1f}%)")
        save_route_csv(result, "optimised_route.csv")
        print("Tip: set use_weather=True to route around real weather conditions")