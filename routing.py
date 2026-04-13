import numpy as np
import pandas as pd
import joblib
import requests
import heapq
import time
import json
import os
from datetime import datetime, timedelta
from shapely.geometry import Point, shape, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_FILE       = "fuel_model.joblib"
SHIP_SPEED_KNOTS = 12.0
GRID_RES         = 0.5  # 0.5 deg (~55 nm) — needed to resolve Red Sea, Malacca

# Mandatory waypoints through chokepoints that are too narrow for the grid.
# A* will be forced to route through these in sequence if they lie between
# start and end (checked by longitude order along the route).
CHOKEPOINTS = [
    ("malacca",  2.5,  101.0),   # Strait of Malacca
    ("hormuz",  26.5,   56.5),   # Strait of Hormuz
    ("bab",     12.5,   43.5),   # Bab-el-Mandeb (Red Sea south entry)
    ("suez_s",  29.9,   32.6),   # Suez Canal south
    ("suez_n",  31.2,   32.4),   # Suez Canal north
]

CHOKEPOINT_REGIONS = [
    # Persian Gulf: needs Hormuz to enter/exit
    ("persian_gulf",    22.0, 31.0,  47.0,  60.0, ["hormuz"]),
    # Red Sea corridor: needs Bab+Suez when one end is in Red Sea
    ("red_sea",         12.0, 30.5,  32.0,  44.0, ["bab", "suez_s", "suez_n"]),
    # Europe/Med/Atlantic: needs Bab+Suez when other end is east of Suez
    ("europe_atlantic", 30.0, 72.0, -80.0,  30.0, ["bab", "suez_s", "suez_n"]),
    # Far East (east of Malacca): needs Malacca when other end is west of it
    ("far_east",         1.0, 45.0, 103.0, 150.0, ["malacca"]),
]

# Coastal penalty: cells within this many grid steps of land pay a cost multiplier.
# 8 steps at 0.5 deg = 4 degrees = ~240 nm buffer around all coastlines.
COAST_PENALTY_STEPS    = 8
COAST_PENALTY_FACTOR   = 5.0   # coastal cells cost 5x open-ocean cells
LAND_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
LAND_CACHE_FILE  = "ne_land_10m.geojson"

# ─────────────────────────────────────────────
# LAND MASK & OPTIMIZATION
# ─────────────────────────────────────────────

def load_land_mask():
    if not os.path.exists(LAND_CACHE_FILE):
        print("  Downloading land mask (~10 MB, once only)...")
        r = requests.get(LAND_GEOJSON_URL, timeout=60)
        r.raise_for_status()
        with open(LAND_CACHE_FILE, "w") as f:
            f.write(r.text)

    with open(LAND_CACHE_FILE) as f:
        gj = json.load(f)

    polygons = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"])
        geom = geom.simplify(0.05, preserve_topology=True)
        if geom.geom_type == "MultiPolygon":
            polygons.extend(geom.geoms)
        else:
            polygons.append(geom)

    # Two trees:
    # - point_tree / point_polys: full polygons for centre-point containment test
    # - edge_tree  / edge_polys:  polygons eroded by 0.1 deg so that a segment
    #   merely grazing a coastline vertex does NOT count as crossing land.
    edge_polys = [p.buffer(-0.1) for p in polygons]
    edge_polys = [p for p in edge_polys if not p.is_empty]

    point_tree = STRtree(polygons)
    edge_tree  = STRtree(edge_polys)
    print(f"  Land mask ready: {len(polygons)} polygons.")
    return polygons, point_tree, edge_polys, edge_tree


def _intersects_land(polygons, tree, geom):
    """True if geom intersects any land polygon (used for point-in-polygon)."""
    for idx in tree.query(geom):
        if polygons[idx].intersects(geom):
            return True
    return False


def _edge_crosses_land(edge_polys, edge_tree, line):
    """True if line crosses the eroded land interior (ignores coastline grazes)."""
    for idx in edge_tree.query(line):
        if edge_polys[idx].intersects(line):
            return True
    return False


def build_ocean_mask(grid, polygons, point_tree):
    print("  Building ocean mask...", end=" ", flush=True)
    mask = np.ones((grid.n_rows, grid.n_cols), dtype=bool)
    for r in range(grid.n_rows):
        if r % 10 == 0: print(f"{int(r/grid.n_rows*100)}%..", end="", flush=True)
        for c in range(grid.n_cols):
            lat, lon = grid.latlon(r, c)
            if _intersects_land(polygons, point_tree, Point(lon, lat)):
                mask[r, c] = False

    # Distance-transform penalty: BFS outward from every land cell.
    # Any ocean cell within COAST_PENALTY_STEPS of land gets COAST_PENALTY_FACTOR.
    # Uses only the boolean mask — zero extra geometry queries.
    dist = np.full((grid.n_rows, grid.n_cols), np.inf)
    from collections import deque
    q = deque()
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            if not mask[r, c]:
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        if dist[r, c] >= COAST_PENALTY_STEPS:
            continue
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if grid.valid(nr, nc) and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

    penalty = np.where((mask) & (dist <= COAST_PENALTY_STEPS),
                       COAST_PENALTY_FACTOR, 1.0)
    print(" done.")
    return mask, penalty

# ─────────────────────────────────────────────
# GEOMETRY & UTILS
# ─────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
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

class Grid:
    def __init__(self, lat_min, lat_max, lon_min, lon_max, res):
        self.res = res
        self.lat_min, self.lon_min = lat_min, lon_min
        self.n_rows = int(round((lat_max - lat_min) / res)) + 1
        self.n_cols = int(round((lon_max - lon_min) / res)) + 1

    def latlon(self, r, c):
        return (round(self.lat_min + r * self.res, 6), round(self.lon_min + c * self.res, 6))

    def rowcol(self, lat, lon):
        r = int(round((lat - self.lat_min) / self.res))
        c = int(round((lon - self.lon_min) / self.res))
        return (max(0, min(r, self.n_rows-1)), max(0, min(c, self.n_cols-1)))

    def valid(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def neighbors(self, r, c, ocean_mask):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if self.valid(nr, nc) and ocean_mask[nr, nc]:
                    yield nr, nc

# ─────────────────────────────────────────────
# FUEL PREDICTION
# ─────────────────────────────────────────────

def predict_fuel_rate(model, features, speed, heading, weather):
    rw = rel_wind(heading, weather.get("wind_dir_deg", 0))
    row = {
        "sog": speed, "stw_knots": speed,
        "wave_height_m": weather.get("wave_height_m", 1.0),
        "wave_period_s": weather.get("wave_period_s", 8.0),
        "wind_speed_ms": weather.get("wind_speed_ms", 5.0),
        "rel_wind_angle_deg": rw, "current_u_ms": 0.0, "current_v_ms": 0.0,
    }
    X = np.array([row.get(f, 0) for f in features]).reshape(1, -1)
    return float(model.predict(X)[0])

# ─────────────────────────────────────────────
# A* ROUTER
# ─────────────────────────────────────────────

def _snap_to_ocean(grid, ocean_mask, lat, lon):
    r, c = grid.rowcol(lat, lon)
    if ocean_mask[r, c]: return r, c
    for radius in range(1, 20):
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                if abs(dr) != radius and abs(dc) != radius: continue
                nr, nc = r+dr, c+dc
                if grid.valid(nr, nc) and ocean_mask[nr, nc]: return nr, nc
    return r, c


def _astar_leg(start_latlon, end_latlon, model, features, departure_time,
               ocean_mask, coast_penalty, polygons, point_tree, edge_polys, edge_tree, grid, speed):
    calm = {"wave_height_m": 1.0, "wave_period_s": 8.0, "wind_speed_ms": 5.0, "wind_dir_deg": 0}
    min_rate = predict_fuel_rate(model, features, speed, 0, calm)

    sr, sc = _snap_to_ocean(grid, ocean_mask, *start_latlon)
    er, ec = _snap_to_ocean(grid, ocean_mask, *end_latlon)
    print(f"    {grid.latlon(sr,sc)} -> {grid.latlon(er,ec)}")

    edge_cache = {}   # (r,c,nr,nc) -> bool (True = blocked)

    def edge_blocked(r, c, nr, nc):
        key = (r, c, nr, nc)
        if key not in edge_cache:
            la,  lo  = grid.latlon(r,  c)
            nla, nlo = grid.latlon(nr, nc)
            edge_cache[key] = _edge_crosses_land(
                edge_polys, edge_tree, LineString([(lo, la), (nlo, nla)]))
        return edge_cache[key]

    def h(r, c):
        la, lo = grid.latlon(r, c)
        ea, eo = grid.latlon(er, ec)
        return min_rate * haversine_nm(la, lo, ea, eo) / speed

    heap = [(h(sr, sc), 0.0, 0.0, sr, sc, [(sr, sc)], departure_time)]
    best_g = {}
    best_g_actual = {}
    iters = 0

    while heap and iters < 2_000_000:
        f, g, g_actual, r, c, path, t = heapq.heappop(heap)
        iters += 1
        if best_g.get((r, c), float("inf")) <= g: continue
        best_g[(r, c)] = g
        best_g_actual[(r, c)] = g_actual

        if r == er and c == ec:
            return [grid.latlon(pr, pc) for pr, pc in path], g_actual, t

        for nr, nc in grid.neighbors(r, c, ocean_mask):
            if edge_blocked(r, c, nr, nc): continue
            la, lo   = grid.latlon(r,  c)
            nla, nlo = grid.latlon(nr, nc)
            dist = haversine_nm(la, lo, nla, nlo)
            hrs  = dist / speed
            rate = predict_fuel_rate(model, features, speed, bearing_deg(la, lo, nla, nlo), calm)
            actual_cost = rate * hrs
            # Penalty steers A* away from coasts but is NOT counted in reported fuel
            routing_cost = actual_cost * float(coast_penalty[nr, nc])
            new_g        = g + routing_cost
            new_g_actual = best_g_actual.get((r, c), 0.0) + actual_cost
            if best_g.get((nr, nc), float("inf")) > new_g:
                heapq.heappush(heap, (new_g + h(nr, nc), new_g, new_g_actual, nr, nc,
                                      path + [(nr, nc)], t + timedelta(hours=hrs)))

    print(f"    WARNING: no path found after {iters} iters")
    return None, 0, departure_time


def astar_route(start_latlon, end_latlon, model, features, departure_time,
                land_geometry, speed=SHIP_SPEED_KNOTS, grid_res=GRID_RES):
    polygons, point_tree, edge_polys, edge_tree = land_geometry

    # Include a chokepoint only when one endpoint is inside a region that
    # requires it AND the other endpoint is outside that region.
    choke_lookup = {name: (lat, lon) for name, lat, lon in CHOKEPOINTS}
    needed = set()
    s_lat, s_lon = start_latlon
    e_lat, e_lon = end_latlon
    for region_name, rlat_min, rlat_max, rlon_min, rlon_max, req in CHOKEPOINT_REGIONS:
        s_in = rlat_min <= s_lat <= rlat_max and rlon_min <= s_lon <= rlon_max
        e_in = rlat_min <= e_lat <= rlat_max and rlon_min <= e_lon <= rlon_max
        if s_in != e_in:   # exactly one endpoint inside the region
            needed.update(req)

    # Order chokepoints by longitude in the direction of travel
    choke_pts = [(name, choke_lookup[name]) for name in needed if name in choke_lookup]
    choke_pts.sort(key=lambda x: x[1][1], reverse=(s_lon > e_lon))
    waypoints = [start_latlon] + [ll for _, ll in choke_pts] + [end_latlon]
    print(f"Waypoint sequence: {' -> '.join(str(w) for w in waypoints)}")

    # One grid covering the full voyage
    all_lats = [w[0] for w in waypoints]
    all_lons = [w[1] for w in waypoints]
    pad = 8.0
    grid = Grid(min(all_lats)-pad, max(all_lats)+pad,
                min(all_lons)-pad, max(all_lons)+pad, grid_res)
    print(f"Grid: {grid.n_rows}x{grid.n_cols} = {grid.n_rows*grid.n_cols} cells")

    ocean_mask, coast_penalty = build_ocean_mask(grid, polygons, point_tree)

    full_route, total_fuel, t = [], 0.0, departure_time
    for i in range(len(waypoints) - 1):
        print(f"  Leg {i+1}/{len(waypoints)-1}:")
        leg, fuel, t = _astar_leg(waypoints[i], waypoints[i+1], model, features,
                                   t, ocean_mask, coast_penalty, polygons, point_tree,
                                   edge_polys, edge_tree, grid, speed)
        if leg is None:
            print(f"  No path for leg {i+1} — aborting.")
            return {}
        full_route += leg if i == 0 else leg[1:]
        total_fuel += fuel

    route_dist = sum(haversine_nm(full_route[i][0], full_route[i][1],
                                  full_route[i+1][0], full_route[i+1][1])
                     for i in range(len(full_route)-1))
    return {"route": full_route,
            "total_fuel_tonnes": round(total_fuel, 3),
            "total_distance_nm": round(route_dist, 1)}


# ─────────────────────────────────────────────
# PORT REGISTRY
# ─────────────────────────────────────────────

PORTS = {
    # East Asia
     1: ("Shanghai",        31.2,  121.5),
     2: ("Singapore",        1.3,  103.8),
     3: ("Hong Kong",       22.3,  114.2),
     4: ("Busan",           35.1,  129.0),
     5: ("Tokyo / Yokohama",35.4,  139.6),
     6: ("Guangzhou",       22.6,  113.6),
    # South / Southeast Asia
     7: ("Mumbai",          18.9,   72.8),
     8: ("Colombo",          6.9,   79.9),
     9: ("Port Klang",       3.0,  101.4),
    10: ("Jakarta",         -6.1,  106.8),
    # Middle East
    11: ("Kuwait",          29.3,   47.9),
    12: ("Dubai / Jebel Ali",24.9,  55.1),
    13: ("Oman / Sohar",    24.3,   56.6),
    # East Africa
    14: ("Mombasa",         -4.1,   39.7),
    15: ("Dar es Salaam",   -6.8,   39.3),
    # Europe
    16: ("Rotterdam",       51.9,    4.5),
    17: ("Hamburg",         53.5,    9.9),
    18: ("Antwerp",         51.3,    4.4),
    19: ("Barcelona",       41.4,    2.2),
    20: ("Piraeus",         37.9,   23.6),
    # Mediterranean / Black Sea
    21: ("Istanbul",        41.0,   28.9),
    22: ("Alexandria",      31.2,   29.9),
    # Americas
    23: ("New York",        40.7,  -74.0),
    24: ("Los Angeles",     33.7, -118.2),
    25: ("Houston",         29.7,  -95.0),
    26: ("Santos",         -23.9,  -46.3),
    27: ("Vancouver",       49.3, -123.1),
    # Africa
    28: ("Durban",         -29.9,   31.0),
    29: ("Cape Town",      -33.9,   18.4),
    30: ("Lagos",            6.4,    3.4),
    # Australia
    31: ("Sydney",         -33.9,  151.2),
    32: ("Melbourne",      -37.8,  144.9),
}


def select_port(prompt):
    print(f"\n{'─'*48}")
    print(f"  {prompt}")
    print(f"{'─'*48}")
    cols = 2
    items = list(PORTS.items())
    half = (len(items) + 1) // 2
    for i in range(half):
        left  = f"  {items[i][0]:>2}. {items[i][1][0]}"
        right = f"  {items[i+half][0]:>2}. {items[i+half][1][0]}" if i+half < len(items) else ""
        print(f"{left:<30}{right}")
    print(f"{'─'*48}")
    while True:
        raw = input("  Enter number: ").strip()
        if raw.isdigit() and int(raw) in PORTS:
            name, lat, lon = PORTS[int(raw)]
            print(f"  Selected: {name} ({lat}, {lon})")
            return name, (lat, lon)
        print(f"  Invalid — enter a number between 1 and {max(PORTS)}.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    def load_model(path):
        saved = joblib.load(path)
        return saved["model"], saved["features"]

    model, features = load_model(MODEL_FILE)
    land = load_land_mask()

    origin_name,      origin_ll = select_port("Select ORIGIN port")
    destination_name, dest_ll   = select_port("Select DESTINATION port")

    if origin_ll == dest_ll:
        print("Origin and destination are the same port. Exiting.")
        exit()

    raw_date = input("\n  Departure date (YYYY-MM-DD) [default: today]: ").strip()
    try:
        departure = datetime.strptime(raw_date, "%Y-%m-%d")
    except ValueError:
        departure = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"  Using today: {departure.strftime('%Y-%m-%d')}")

    print(f"\nRouting {origin_name} -> {destination_name} | Departure {departure.strftime('%Y-%m-%d')}")

    result = astar_route(origin_ll, dest_ll, model, features, departure, land)

    if result:
        print(f"\nRoute Found!")
        print(f"  Distance : {result['total_distance_nm']} nm")
        print(f"  Fuel     : {result['total_fuel_tonnes']} t")
        slug = f"{origin_name.split('/')[0].strip().lower().replace(' ', '_')}_to_{destination_name.split('/')[0].strip().lower().replace(' ', '_')}"
        out  = f"optimised_route_{slug}.csv"
        pd.DataFrame(result['route'], columns=['lat', 'lon']).to_csv(out, index=False)
        print(f"  Saved    : {out}")