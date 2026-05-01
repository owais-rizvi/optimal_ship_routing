import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routing import load_land_mask, astar_route, PORTS, MODEL_FILE, EXCLUSION_ZONES
from datetime import datetime
import numpy as np

app = FastAPI(title="Ship Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, features = None, None
land = None

@app.on_event("startup")
def startup_event():
    global model, features, land
    print("Loading model...")
    saved = joblib.load(MODEL_FILE)
    model, features = saved["model"], saved["features"]
    print("Loading land mask...")
    land = load_land_mask()
    print("API is ready.")

@app.get("/ports")
def get_ports():
    return [{"id": k, "name": v[0], "lat": v[1], "lon": v[2]} for k, v in PORTS.items()]

@app.get("/exclusion_zones")
def get_exclusion_zones():
    return EXCLUSION_ZONES

@app.get("/route")
def get_route(origin: int, dest: int, date: str = None, speed: float = 12.0):
    if origin not in PORTS or dest not in PORTS:
        raise HTTPException(status_code=400, detail="Invalid port ID")
    
    origin_name = PORTS[origin][0]
    dest_name = PORTS[dest][0]
    origin_ll = (PORTS[origin][1], PORTS[origin][2])
    dest_ll = (PORTS[dest][1], PORTS[dest][2])
    
    if origin_ll == dest_ll:
        raise HTTPException(status_code=400, detail="Origin and destination cannot be the same")
        
    origin_slug = origin_name.split('/')[0].strip().lower().replace(' ', '_')
    dest_slug = dest_name.split('/')[0].strip().lower().replace(' ', '_')
    
    filename_fwd = f"optimised_route_{origin_slug}_to_{dest_slug}.csv"
    filename_rev = f"optimised_route_{dest_slug}_to_{origin_slug}.csv"
    
    import os
    import pandas as pd
    from routing import haversine_nm, bearing_deg, predict_fuel_rate
    
    calm = {"wave_height_m": 1.0, "wave_period_s": 8.0, "wind_speed_ms": 5.0, "wind_dir_deg": 0}
    
    # Baseline calculations
    baseline_route = [{"lat": float(origin_ll[0]), "lon": float(origin_ll[1])}, {"lat": float(dest_ll[0]), "lon": float(dest_ll[1])}]
    d_baseline = haversine_nm(origin_ll[0], origin_ll[1], dest_ll[0], dest_ll[1])
    brg_baseline = bearing_deg(origin_ll[0], origin_ll[1], dest_ll[0], dest_ll[1])
    rate_baseline = predict_fuel_rate(model, features, speed, brg_baseline, calm)
    fuel_baseline = rate_baseline * (d_baseline / speed)
    
    target_filename = None
    is_reversed = False
    
    if os.path.exists(filename_fwd):
        target_filename = filename_fwd
    elif os.path.exists(filename_rev):
        target_filename = filename_rev
        is_reversed = True
        
    if target_filename:
        print(f"Loading cached route from {target_filename} (reversed: {is_reversed})")
        df = pd.read_csv(target_filename)
        route_coords = []
        route_list = []
        for _, row in df.iterrows():
            try:
                lat = float(row['lat'])
                lon = float(row['lon'])
                route_coords.append({"lat": lat, "lon": lon})
                route_list.append((lat, lon))
            except ValueError:
                pass
                
        if is_reversed:
            route_coords.reverse()
            route_list.reverse()
                
        dist = 0.0
        fuel = 0.0
        for i in range(len(route_list)-1):
            la, lo = route_list[i]
            nla, nlo = route_list[i+1]
            d = haversine_nm(la, lo, nla, nlo)
            dist += d
            brg = bearing_deg(la, lo, nla, nlo)
            rate = predict_fuel_rate(model, features, speed, brg, calm)
            fuel += rate * (d / speed)
            
        return {
            "route": route_coords,
            "total_fuel_tonnes": round(fuel, 3),
            "total_distance_nm": round(dist, 1),
            "baseline_route": baseline_route,
            "baseline_fuel_tonnes": round(fuel_baseline, 3),
            "baseline_distance_nm": round(d_baseline, 1)
        }

    try:
        if date:
            departure = datetime.strptime(date, "%Y-%m-%d")
        else:
            departure = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")
        
    # We pass the custom speed to the router by mocking the import or we can just use astar_route with speed
    result = astar_route(origin_ll, dest_ll, model, features, departure, land, speed=speed)
    if result:
        route_coords = [{"lat": float(p[0]), "lon": float(p[1])} for p in result["route"]]
        
        # Save to cache so next time we load it
        pd.DataFrame([{"lat": p["lat"], "lon": p["lon"]} for p in route_coords]).to_csv(filename_fwd, index=False)
        
        return {
            "route": route_coords,
            "total_fuel_tonnes": result["total_fuel_tonnes"],
            "total_distance_nm": result["total_distance_nm"],
            "baseline_route": baseline_route,
            "baseline_fuel_tonnes": round(fuel_baseline, 3),
            "baseline_distance_nm": round(d_baseline, 1)
        }
    else:
        raise HTTPException(status_code=404, detail="No route found")
