"""
NavOptima Backend
=================
Flask API that bridges the HTML frontend to the existing
routing.py A* engine + fuel model.

Install:
    pip install flask flask-cors

Run:
    python app.py
    # Then open http://localhost:5000 in your browser
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, json
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Try to load the real ML model + routing engine ──
MODEL_LOADED = False
model, features, land = None, None, None

try:
    sys.path.insert(0, os.path.dirname(__file__))
    from routing import load_model, load_land_mask, astar_route, great_circle_fuel, SHIP_SPEED_KNOTS

    MODEL_FILE = os.path.join(os.path.dirname(__file__), "fuel_model.joblib")
    if os.path.exists(MODEL_FILE):
        print("[NavOptima] Loading XGBoost fuel model…")
        model, features = load_model(MODEL_FILE)
        print("[NavOptima] Loading land mask…")
        land = load_land_mask()
        MODEL_LOADED = True
        print("[NavOptima] ✓ ML engine ready")
    else:
        print(f"[NavOptima] ⚠ fuel_model.joblib not found at {MODEL_FILE}")
        print("[NavOptima] → Frontend will use built-in Admiralty formula simulation")

except ImportError as e:
    print(f"[NavOptima] ⚠ Could not import routing.py: {e}")
    print("[NavOptima] → Frontend will use built-in Admiralty formula simulation")


# ── Serve the frontend ──
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── Health / status endpoint ──
@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "engine": "XGBoost + A*" if MODEL_LOADED else "Admiralty formula (simulation)",
        "features": features if MODEL_LOADED else None,
    })


# ── Main routing endpoint ──
@app.route("/api/route", methods=["POST"])
def compute_route():
    data = request.get_json()

    origin_lat  = float(data["origin_lat"])
    origin_lon  = float(data["origin_lon"])
    dest_lat    = float(data["dest_lat"])
    dest_lon    = float(data["dest_lon"])
    speed       = float(data.get("speed", 12.0))
    grid_res    = float(data.get("grid_res", 1.0))
    use_weather = bool(data.get("use_weather", False))
    departure   = data.get("departure", "2024-06-15")

    dep_dt = datetime.strptime(departure, "%Y-%m-%d")

    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Train the model first (train_model.py)."}), 503

    try:
        # ── Great circle baseline ──
        gc = great_circle_fuel(
            start=(origin_lat, origin_lon),
            end=(dest_lat, dest_lon),
            model=model,
            features=features,
            departure_time=dep_dt,
            speed=speed,
        )

        # ── Optimised route ──
        result = astar_route(
            start_latlon=(origin_lat, origin_lon),
            end_latlon=(dest_lat, dest_lon),
            model=model,
            features=features,
            departure_time=dep_dt,
            land_geometry=land,
            speed=speed,
            grid_res=grid_res,
            use_weather=use_weather,
        )

        if not result:
            return jsonify({"error": "No route found. Try increasing grid resolution or check port coordinates."}), 422

        # Serialize
        travel_hrs = (result["eta"] - dep_dt).total_seconds() / 3600
        saving = gc["total_fuel_tonnes"] - result["total_fuel_tonnes"]
        saving_pct = round(saving / gc["total_fuel_tonnes"] * 100, 1) if gc["total_fuel_tonnes"] > 0 else 0

        return jsonify({
            "route":              [[float(p[0]), float(p[1])] for p in result["route"]],
            "gc_route":           [[float(p[0]), float(p[1])] for p in gc["route"]],
            "opt_fuel":           result["total_fuel_tonnes"],
            "opt_dist":           result["total_distance_nm"],
            "gc_fuel":            gc["total_fuel_tonnes"],
            "gc_dist":            gc["total_distance_nm"],
            "travel_hrs":         round(travel_hrs, 1),
            "eta":                result["eta"].strftime("%Y-%m-%d %H:%M UTC"),
            "fuel_saving_tonnes": round(saving, 2),
            "fuel_saving_pct":    saving_pct,
            "waypoint_count":     len(result["route"]),
            "engine":             "XGBoost + A* (real model)",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Fuel prediction endpoint (single point) ──
@app.route("/api/predict_fuel", methods=["POST"])
def predict_fuel():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    d = request.get_json()
    import numpy as np
    row = {
        "sog": float(d.get("sog", 12)),
        "stw_knots": float(d.get("stw_knots", 11.5)),
        "wave_height_m": float(d.get("wave_height_m", 1.0)),
        "wave_period_s": float(d.get("wave_period_s", 8.0)),
        "wind_speed_ms": float(d.get("wind_speed_ms", 5.0)),
        "rel_wind_angle_deg": float(d.get("rel_wind_angle_deg", 90)),
        "current_u_ms": float(d.get("current_u_ms", 0)),
        "current_v_ms": float(d.get("current_v_ms", 0)),
    }
    X = np.array([row.get(f, 0) for f in features]).reshape(1, -1)
    fuel_rate = float(model.predict(X)[0])

    return jsonify({
        "fuel_rate_tph": round(fuel_rate, 4),
        "fuel_daily_tonnes": round(fuel_rate * 24, 2),
        "inputs": row,
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  NavOptima Ship Routing Frontend")
    print("="*50)
    print(f"  Engine: {'✓ XGBoost + A* (real model)' if MODEL_LOADED else '⚠ Admiralty simulation (no model)'}")
    print(f"  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)