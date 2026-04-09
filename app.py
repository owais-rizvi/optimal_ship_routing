"""
Flask API for the ship routing frontend.
Run:  python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import json, os

from routing import (
    load_model, load_land_mask,
    astar_route, great_circle_fuel,
    MODEL_FILE, SHIP_SPEED_KNOTS, LAND_CACHE_FILE,
)

app = Flask(__name__)

# Load once at startup
print("Loading model...")
model, features = load_model(MODEL_FILE)
print("Loading land mask...")
land = load_land_mask()
print("Ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/land")
def api_land():
    with open(LAND_CACHE_FILE) as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.route("/api/route", methods=["POST"])
def api_route():
    data = request.json
    try:
        start = (float(data["start_lat"]), float(data["start_lon"]))
        end   = (float(data["end_lat"]),   float(data["end_lon"]))
        speed = float(data.get("speed", SHIP_SPEED_KNOTS))
        grid_res     = float(data.get("grid_res", 2.0))
        use_weather  = bool(data.get("use_weather", False))
        departure    = datetime.fromisoformat(data.get("departure", "2026-01-01T00:00"))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    gc = great_circle_fuel(start, end, model, features, departure, speed=speed)
    opt = astar_route(
        start_latlon=start, end_latlon=end,
        model=model, features=features,
        departure_time=departure, land_geometry=land,
        speed=speed, grid_res=grid_res, use_weather=use_weather,
    )

    if not opt:
        return jsonify({"error": "No route found. Try a coarser grid or different ports."}), 500

    saving    = gc["total_fuel_tonnes"] - opt["total_fuel_tonnes"]
    saving_pct = round(saving / gc["total_fuel_tonnes"] * 100, 1)

    return jsonify({
        "optimised": {
            "route":              opt["route"],
            "total_fuel_tonnes":  opt["total_fuel_tonnes"],
            "total_distance_nm":  opt["total_distance_nm"],
            "eta":                opt["eta"].isoformat(),
        },
        "great_circle": {
            "route":              [list(p) for p in gc["route"]],
            "total_fuel_tonnes":  gc["total_fuel_tonnes"],
            "total_distance_nm":  gc["total_distance_nm"],
            "eta":                gc["eta"].isoformat(),
        },
        "fuel_saving_tonnes": round(saving, 3),
        "fuel_saving_pct":    saving_pct,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
