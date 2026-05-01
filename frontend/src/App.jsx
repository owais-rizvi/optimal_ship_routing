import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap, Polygon } from 'react-leaflet';
import L from 'leaflet';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const API_BASE = 'http://localhost:8000';

function ChangeView({ bounds }) {
  const map = useMap();
  if (bounds) {
    map.fitBounds(bounds, { padding: [50, 50] });
  }
  return null;
}

function App() {
  const [ports, setPorts] = useState([]);
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [speed, setSpeed] = useState(12);
  const [exclusionZones, setExclusionZones] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [routeData, setRouteData] = useState(null);
  const [error, setError] = useState('');

  // Fetch ports on load
  useEffect(() => {
    axios.get(`${API_BASE}/ports`)
      .then(res => {
        setPorts(res.data.sort((a, b) => a.name.localeCompare(b.name)));
      })
      .catch(err => {
        console.error("Failed to load ports", err);
        setError("Failed to connect to the backend API.");
      });
      
    axios.get(`${API_BASE}/exclusion_zones`)
      .then(res => setExclusionZones(res.data))
      .catch(err => console.error("Failed to load exclusion zones", err));
  }, []);

  const handleRoute = async (e) => {
    e.preventDefault();
    if (!origin || !destination) {
      setError("Please select both origin and destination ports.");
      return;
    }
    if (origin === destination) {
      setError("Origin and destination cannot be the same.");
      return;
    }

    setLoading(true);
    setError('');
    setRouteData(null);

    try {
      const res = await axios.get(`${API_BASE}/route`, {
        params: {
          origin,
          dest: destination,
          date,
          speed
        }
      });
      setRouteData(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred while calculating the route.");
    } finally {
      setLoading(false);
    }
  };

  const getPortCoordinates = (id) => {
    const port = ports.find(p => p.id.toString() === id.toString());
    return port ? [port.lat, port.lon] : null;
  };

  // Convert API route format to Leaflet latlng array
  const routePositions = routeData ? routeData.route.map(p => [p.lat, p.lon]) : [];
  const baselinePositions = routeData?.baseline_route ? routeData.baseline_route.map(p => [p.lat, p.lon]) : [];
  
  // Calculate bounds to fit the route
  const bounds = routePositions.length > 0 ? L.latLngBounds(routePositions) : null;

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="header">
          <h1>Ocean Route Optimizer</h1>
          <p>Machine Learning powered voyage optimization for minimum fuel consumption and weather avoidance.</p>
        </div>

        <form onSubmit={handleRoute} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="form-group">
            <label>Origin Port</label>
            <select value={origin} onChange={(e) => setOrigin(e.target.value)}>
              <option value="">-- Select Origin --</option>
              {ports.map(p => (
                <option key={`org-${p.id}`} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Destination Port</label>
            <select value={destination} onChange={(e) => setDestination(e.target.value)}>
              <option value="">-- Select Destination --</option>
              {ports.map(p => (
                <option key={`dest-${p.id}`} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Departure Date</label>
            <input 
              type="date" 
              value={date} 
              onChange={(e) => setDate(e.target.value)} 
            />
          </div>

          <div className="form-group">
            <label>Vessel Speed: {speed} knots</label>
            <input 
              type="range" 
              min="8" 
              max="24" 
              step="0.5"
              value={speed} 
              onChange={(e) => setSpeed(parseFloat(e.target.value))} 
            />
          </div>

          <button type="submit" disabled={loading || !ports.length}>
            {loading ? <div className="loader"></div> : "Calculate Optimal Route"}
          </button>
        </form>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {routeData && (
          <div className="results-panel">
            <h3>Voyage Summary</h3>
            <div className="stat-row">
              <span className="label">Optimized Fuel</span>
              <span className="value">{routeData.total_fuel_tonnes.toLocaleString()} t</span>
            </div>
            <div className="stat-row">
              <span className="label">Baseline Fuel (Straight Line)</span>
              <span className="value">{routeData.baseline_fuel_tonnes.toLocaleString()} t</span>
            </div>
            <div className="stat-row" style={{ color: 'var(--success)', fontWeight: 'bold' }}>
              <span className="label">Fuel Saved</span>
              <span className="value">{(routeData.baseline_fuel_tonnes - routeData.total_fuel_tonnes).toFixed(1)} t</span>
            </div>
            <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '1rem 0' }} />
            <div className="stat-row">
              <span className="label">Optimized Distance</span>
              <span className="value">{routeData.total_distance_nm.toLocaleString()} nm</span>
            </div>
            <div className="stat-row">
              <span className="label">Waypoints</span>
              <span className="value">{routeData.route.length} nodes</span>
            </div>
          </div>
        )}
      </div>

      <div className="map-container">
        <MapContainer 
          center={[20, 0]} 
          zoom={2} 
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {bounds && <ChangeView bounds={bounds} />}

          {origin && getPortCoordinates(origin) && (
            <Marker position={getPortCoordinates(origin)}>
              <Popup>Origin: {ports.find(p => p.id.toString() === origin.toString())?.name}</Popup>
            </Marker>
          )}

          {destination && getPortCoordinates(destination) && (
            <Marker position={getPortCoordinates(destination)}>
              <Popup>Destination: {ports.find(p => p.id.toString() === destination.toString())?.name}</Popup>
            </Marker>
          )}

          {exclusionZones.map((zone, idx) => (
            <Polygon 
              key={`zone-${idx}`} 
              positions={zone.coordinates} 
              pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.2, weight: 1 }}
            >
              <Popup><strong>Exclusion Zone</strong><br/>{zone.name}</Popup>
            </Polygon>
          ))}

          {baselinePositions.length > 0 && (
            <Polyline 
              positions={baselinePositions} 
              pathOptions={{ 
                color: '#ef4444', 
                weight: 2, 
                opacity: 0.6,
                dashArray: '5, 10',
                lineCap: 'round'
              }} 
            />
          )}

          {routePositions.length > 0 && (
            <Polyline 
              positions={routePositions} 
              pathOptions={{ 
                color: '#3b82f6', 
                weight: 4, 
                opacity: 0.8,
                dashArray: '10, 10',
                lineCap: 'round'
              }} 
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
