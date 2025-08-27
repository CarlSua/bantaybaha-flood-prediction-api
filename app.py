# app.py - Simple Flood Prediction API
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow web browsers to access the API

print("🚀 Starting Flood Prediction API...")

# Load the trained model and components
try:
    print("📦 Loading model files...")
    model = joblib.load('realtime_flood_model.pkl')
    scaler = joblib.load('realtime_scaler.pkl')
    config = joblib.load('realtime_config.pkl')
    
    # Get thresholds from config
    water_level_threshold = config['water_level_threshold']
    water_pressure_threshold = config['water_pressure_threshold']
    rain_threshold = config['rain_threshold']
    
    print("✅ Model loaded successfully!")
    print(f"🎯 Thresholds - Water: {water_level_threshold:.2f}m, Pressure: {water_pressure_threshold:.2f}kPa, Rain: {rain_threshold:.2f}mm")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have run the Jupyter notebook first!")
    model = None
    scaler = None
    config = None

# Basic API metadata
API_VERSION = "1.0.0"

REQUIRED_FIELDS = {
    "water_level_m": (float, "Water level in meters"),
    "water_pressure_kpa": (float, "Water pressure in kPa"),
    "rain_precip_mm": (float, "Rain precipitation in mm"),
}

def _validate_and_cast_payload(json_data):
    """Validate incoming JSON and cast to floats. Returns (payload_dict, error_message)."""
    if not isinstance(json_data, dict):
        return None, "Invalid JSON body. Expected an object."

    casted = {}
    for field, (cast_type, _desc) in REQUIRED_FIELDS.items():
        if field not in json_data:
            return None, f"Missing required field: {field}"
        try:
            casted[field] = cast_type(json_data[field])
        except Exception:
            return None, f"Field {field} must be a number."
    # Timestamp optional
    ts = json_data.get("timestamp")
    if ts is None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    casted["timestamp"] = str(ts)
    return casted, None

def _build_feature_vector(payload: dict) -> np.ndarray:
    """Build feature vector in the exact training order saved in config['features']."""
    if not config or 'features' not in config:
        raise RuntimeError("Model config missing 'features'. Retrain/export model.")

    # Extract raw inputs
    water_level = float(payload['water_level_m'])
    water_pressure = float(payload['water_pressure_kpa'])
    rain_precip = float(payload['rain_precip_mm'])
    timestamp = pd.to_datetime(payload['timestamp'])

    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    is_weekend = 1 if day_of_week >= 5 else 0

    # Risk indicators from thresholds
    high_water_level = 1 if water_level > water_level_threshold else 0
    high_pressure = 1 if water_pressure > water_pressure_threshold else 0
    heavy_rain = 1 if rain_precip > rain_threshold else 0

    # Interactions
    water_pressure_ratio = water_level * water_pressure
    rain_water_interaction = rain_precip * water_level

    # Map all engineered values
    feature_map = {
        'water_level_m': water_level,
        'water_pressure_kpa': water_pressure,
        'rain_precip_mm': rain_precip,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'high_water_level': high_water_level,
        'high_pressure': high_pressure,
        'heavy_rain': heavy_rain,
        'water_pressure_ratio': water_pressure_ratio,
        'rain_water_interaction': rain_water_interaction,
    }

    # Build in exact order
    ordered = [feature_map[name] for name in config['features']]
    return np.array([ordered])

# Home page - shows API info
@app.route('/')
def home():
    return jsonify({
        "message": "🌊 Flood Prediction API is running!",
        "status": "online" if model else "model not loaded",
        "endpoints": {
            "/predict": "POST - Send sensor data to predict flood",
            "/test": "GET - Test the API with sample data",
            "/health": "GET - Check if API is working",
            "/schema": "GET - Input schema and feature order",
            "/version": "GET - API and model versions"
        },
        "required_data": {
            "water_level_m": "Water level in meters",
            "water_pressure_kpa": "Water pressure in kPa", 
            "rain_precip_mm": "Rain precipitation in mm",
            "timestamp": "Current time (optional)"
        },
        "api_version": API_VERSION
    })

# Main prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not (model and scaler and config):
            return jsonify({"error": "Model not loaded. Run Jupyter notebook first!"}), 500
        
        # Validate payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received. Send JSON data."}), 400

        payload, err = _validate_and_cast_payload(data)
        if err:
            return jsonify({"error": err, "schema": list(REQUIRED_FIELDS.keys())}), 400

        print(f"📡 Received data: Water={payload['water_level_m']}m, Pressure={payload['water_pressure_kpa']}kPa, Rain={payload['rain_precip_mm']}mm")

        # Build features in the exact training order
        features = _build_feature_vector(payload)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        flood_probability = probabilities[1]

        # Rule-based overrides for extreme conditions
        override_reason = None
        wl = float(payload['water_level_m'])
        wp = float(payload['water_pressure_kpa'])
        rn = float(payload['rain_precip_mm'])

        # User-requested thresholds
        USER_WATER_LEVEL_THRESH = 5.0
        USER_WATER_PRESSURE_THRESH = 30.0
        USER_RAIN_THRESH = 60.0

        if wp >= USER_WATER_PRESSURE_THRESH:
            flood_probability = 1.0
            prediction = 1
            override_reason = f"water_pressure_kpa>=${{USER_WATER_PRESSURE_THRESH}}"
        elif rn >= USER_RAIN_THRESH:
            if flood_probability < 0.95:
                flood_probability = 0.95
            prediction = 1
            override_reason = f"rain_precip_mm>=${{USER_RAIN_THRESH}}"
        elif wl >= USER_WATER_LEVEL_THRESH:
            if flood_probability < 0.85:
                flood_probability = 0.85
            prediction = 1
            override_reason = f"water_level_m>=${{USER_WATER_LEVEL_THRESH}}"
        
        # Determine risk level
        if flood_probability < 0.1:
            risk_level = "Very Low"
            alert = "✅ Normal conditions"
            color = "green"
        elif flood_probability < 0.3:
            risk_level = "Low"
            alert = "📋 Monitor conditions"  
            color = "blue"
        elif flood_probability < 0.5:
            risk_level = "Moderate"
            alert = "⚡ Stay alert"
            color = "yellow"
        elif flood_probability < 0.7:
            risk_level = "High"
            alert = "⚠️ Prepare for flooding"
            color = "orange"
        else:
            risk_level = "Critical"
            alert = "🚨 FLOOD ALERT!"
            color = "red"
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": payload['timestamp'],
            "flood_prediction": int(prediction),
            "flood_probability": round(float(flood_probability * 100), 1),  # Convert to percentage
            "risk_level": risk_level,
            "alert_message": alert,
            "color": color,
            "sensor_data": {
                "water_level_m": float(payload['water_level_m']),
                "water_pressure_kpa": float(payload['water_pressure_kpa']),
                "rain_precip_mm": float(payload['rain_precip_mm'])
            },
            "features": config.get('features', []),
            "model_version": API_VERSION,
            "rule_override_applied": bool(override_reason),
            "override_reason": override_reason
        }
        
        print(f"🎯 Prediction: {flood_probability*100:.1f}% risk - {risk_level}")
        return jsonify(response)
        
    except KeyError as e:
        return jsonify({
            "error": f"Missing required field: {str(e)}",
            "required": ["water_level_m", "water_pressure_kpa", "rain_precip_mm"]
        }), 400
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Test endpoint
@app.route('/test', methods=['GET'])
def test():
    """Test the API with sample data"""
    sample_data = {
        "water_level_m": 2.5,
        "water_pressure_kpa": 25.0,
        "rain_precip_mm": 1.0,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Call the predict function internally
    with app.test_client() as client:
        response = client.post('/predict', 
                             json=sample_data,
                             content_type='application/json')
        return response.get_json()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model else "model not loaded",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_loaded": model is not None
    })

# Input schema endpoint
@app.route('/schema', methods=['GET'])
def schema():
    return jsonify({
        "required": {k: v[1] for k, v in REQUIRED_FIELDS.items()},
        "optional": {"timestamp": "ISO datetime string"},
        "feature_order": (config.get('features') if config else None)
    })

# Version endpoint
@app.route('/version', methods=['GET'])
def version():
    return jsonify({
        "api_version": API_VERSION,
        "model_artifacts": {
            "model": bool(model),
            "scaler": bool(scaler),
            "config": bool(config)
        }
    })

# Run the Flask app
if __name__ == '__main__':
    print("🌊 Flood Prediction API is starting...")
    print("📡 Your IoT device can send data to:")
    print("   http://localhost:5000/predict")
    print("🔗 Open browser: http://localhost:5000")
    print("🧪 Test endpoint: http://localhost:5000/test")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)