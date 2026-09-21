import os
import gc
import json
import math
from pathlib import Path

# Reduce TensorFlow startup logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"
UPLOAD_FOLDER = BASE_DIR / "backend" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NeuroVortex API",
    description="AI-Based Cyclone Intelligence and Prediction System",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://neuro-vortex-frontend-phi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATHS
# ============================================================

INSAT_MODEL_PATH = MODEL_FOLDER / "best_insat3d_model.keras"
TRACK_MODEL_PATH = MODEL_FOLDER / "cyclone_track_model.pkl"
HISTORY_TRACK_MODEL_PATH = MODEL_FOLDER / "improved_history_track_model.pkl"
INTENSITY_MODEL_PATH = MODEL_FOLDER / "cyclone_intensity_forecast_model.pkl"


# ============================================================
# BASIC HELPERS
# ============================================================

def kmh_from_knots(knots):
    return round(float(knots) * 1.852, 2)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# IMD CLASSIFICATION
# ============================================================

def get_cyclone_category(wind_kt):
    wind_kt = float(wind_kt)

    if wind_kt < 17:
        return "Low Pressure Area"
    elif wind_kt < 28:
        return "Depression"
    elif wind_kt < 34:
        return "Deep Depression"
    elif wind_kt < 48:
        return "Cyclonic Storm"
    elif wind_kt < 64:
        return "Severe Cyclonic Storm"
    elif wind_kt < 90:
        return "Very Severe Cyclonic Storm"
    elif wind_kt < 120:
        return "Extremely Severe Cyclonic Storm"
    else:
        return "Super Cyclonic Storm"


def get_risk_level(wind_kt):
    wind_kt = float(wind_kt)

    if wind_kt < 34:
        return "LOW"
    elif wind_kt < 64:
        return "MODERATE"
    elif wind_kt < 90:
        return "HIGH"
    else:
        return "VERY HIGH"


# ============================================================
# SATELLITE MODEL
# ============================================================

def predict_insat_intensity(image_path):
    """
    Loads EfficientNetB0 only when prediction is requested.
    This prevents Render startup from loading TensorFlow + model.
    """

    import tensorflow as tf

    model = None

    try:
        if not INSAT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Satellite model not found: {INSAT_MODEL_PATH}"
            )

        model = tf.keras.models.load_model(
            INSAT_MODEL_PATH,
            compile=False
        )

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(
            image_array,
            verbose=0
        )

        normalized_prediction = float(np.asarray(prediction).reshape(-1)[0])

        # Training target range
        min_wind = 25.0
        max_wind = 128.0

        predicted_wind = (
            normalized_prediction *
            (max_wind - min_wind)
            + min_wind
        )

        predicted_wind = float(
            np.clip(predicted_wind, min_wind, max_wind)
        )

        return {
            "intensity_kt": round(predicted_wind, 2),
            "intensity_kmh": kmh_from_knots(predicted_wind),
            "category": get_cyclone_category(predicted_wind),
            "risk": get_risk_level(predicted_wind)
        }

    finally:
        # Release TensorFlow model memory
        del model
        gc.collect()

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass


# ============================================================
# IBTRACS 3-HOUR INTENSITY MODEL
# ============================================================

def predict_ibtracs_intensity(data):
    """
    Predict +3 hour cyclone wind intensity.
    """

    model = None

    try:
        if not INTENSITY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Intensity model not found: {INTENSITY_MODEL_PATH}"
            )

        model = joblib.load(INTENSITY_MODEL_PATH)

        current_lat = safe_float(data["current_lat"])
        current_lon = safe_float(data["current_lon"])
        current_wind = safe_float(data["current_wind"])
        speed = safe_float(data["speed"])
        direction = safe_float(data["direction"])

        direction_rad = math.radians(direction)

        direction_sin = math.sin(direction_rad)
        direction_cos = math.cos(direction_rad)

        features = np.array([[
            current_lat,
            current_lon,
            current_wind,
            speed,
            direction_sin,
            direction_cos
        ]], dtype=np.float32)

        prediction = model.predict(features)

        predicted_wind = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        predicted_wind = max(0.0, predicted_wind)

        return {
            "predicted_wind_kt": round(predicted_wind, 2),
            "predicted_wind_kmh": kmh_from_knots(predicted_wind),
            "category": get_cyclone_category(predicted_wind),
            "risk": get_risk_level(predicted_wind)
        }

    finally:
        del model
        gc.collect()


# ============================================================
# IBTRACS 3-HOUR TRACK MODEL
# ============================================================

def predict_ibtracs_track(data):
    """
    Predict cyclone position after 3 hours.
    """

    model = None

    try:
        if not TRACK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Track model not found: {TRACK_MODEL_PATH}"
            )

        model = joblib.load(TRACK_MODEL_PATH)

        current_lat = safe_float(data["current_lat"])
        current_lon = safe_float(data["current_lon"])
        speed = safe_float(data["speed"])
        direction = safe_float(data["direction"])

        direction_rad = math.radians(direction)

        direction_sin = math.sin(direction_rad)
        direction_cos = math.cos(direction_rad)

        features = np.array([[
            current_lat,
            current_lon,
            speed,
            direction_sin,
            direction_cos
        ]], dtype=np.float32)

        prediction = model.predict(features)

        prediction = np.asarray(prediction).reshape(-1)

        predicted_lat = float(prediction[0])
        predicted_lon = float(prediction[1])

        return {
            "latitude": round(predicted_lat, 4),
            "longitude": round(predicted_lon, 4)
        }

    finally:
        del model
        gc.collect()


# ============================================================
# HISTORY-BASED 3–24 HOUR TRACK MODEL
# ============================================================

def predict_history_track(data):
    """
    Predict cyclone positions at:
    +3h, +6h, +12h, +18h, +24h

    Uses the improved history-based Random Forest model.
    """

    model = None

    try:
        if not HISTORY_TRACK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"History track model not found: "
                f"{HISTORY_TRACK_MODEL_PATH}"
            )

        model = joblib.load(HISTORY_TRACK_MODEL_PATH)

        current_lat = safe_float(data["current_lat"])
        current_lon = safe_float(data["current_lon"])

        lat_3h = safe_float(data["lat_3h_ago"])
        lon_3h = safe_float(data["lon_3h_ago"])

        lat_6h = safe_float(data["lat_6h_ago"])
        lon_6h = safe_float(data["lon_6h_ago"])

        speed = safe_float(data["speed"])
        direction = safe_float(data["direction"])

        direction_rad = math.radians(direction)

        direction_sin = math.sin(direction_rad)
        direction_cos = math.cos(direction_rad)

        # Historical movement features
        lat_change_3h = current_lat - lat_3h
        lon_change_3h = current_lon - lon_3h

        lat_change_6h = current_lat - lat_6h
        lon_change_6h = current_lon - lon_6h

        features = np.array([[
            lat_6h,
            lon_6h,
            lat_3h,
            lon_3h,
            current_lat,
            current_lon,
            speed,
            direction_sin,
            direction_cos,
            lat_change_3h,
            lon_change_3h,
            lat_change_6h,
            lon_change_6h
        ]], dtype=np.float32)

        prediction = model.predict(features)

        prediction = np.asarray(prediction).reshape(-1)

        # Expected order:
        # 3h lat, 3h lon,
        # 6h lat, 6h lon,
        # 12h lat, 12h lon,
        # 18h lat, 18h lon,
        # 24h lat, 24h lon

        if len(prediction) < 10:
            raise ValueError(
                "History track model returned fewer than 10 values."
            )

        return {
            "3h": {
                "latitude": round(float(prediction[0]), 4),
                "longitude": round(float(prediction[1]), 4)
            },
            "6h": {
                "latitude": round(float(prediction[2]), 4),
                "longitude": round(float(prediction[3]), 4)
            },
            "12h": {
                "latitude": round(float(prediction[4]), 4),
                "longitude": round(float(prediction[5]), 4)
            },
            "18h": {
                "latitude": round(float(prediction[6]), 4),
                "longitude": round(float(prediction[7]), 4)
            },
            "24h": {
                "latitude": round(float(prediction[8]), 4),
                "longitude": round(float(prediction[9]), 4)
            }
        }

    finally:
        del model
        gc.collect()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "NeuroVortex",
        "message": "NeuroVortex API is running",
        "status": "success"
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def api_status():
    return {
        "name": "NeuroVortex",
        "message": "NeuroVortex API is running",
        "status": "success"
    }


# ============================================================
# SAMPLE PREDICTION
# ============================================================

@app.get("/api/sample-prediction")
def sample_prediction():

    return {
        "satellite_intensity": {
            "intensity_kt": 57.12,
            "intensity_kmh": 105.79,
            "category": "Severe Cyclonic Storm",
            "risk": "MODERATE"
        },

        "intensity_forecast": {
            "predicted_wind_kt": 70.58,
            "predicted_wind_kmh": 130.72,
            "category": "Very Severe Cyclonic Storm",
            "risk": "HIGH"
        },

        "track_forecast": {
            "3h": {
                "latitude": 10.8209,
                "longitude": 71.8933
            },
            "6h": {
                "latitude": 11.0021,
                "longitude": 71.6803
            },
            "12h": {
                "latitude": 11.3968,
                "longitude": 71.2815
            },
            "18h": {
                "latitude": 11.8015,
                "longitude": 70.8859
            },
            "24h": {
                "latitude": 12.1841,
                "longitude": 70.4954
            }
        },

        "status": "success"
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
async def predict(
    data: str = Form(...),
    image: UploadFile = File(...)
):

    image_path = None

    try:

        # ----------------------------------------------------
        # Parse JSON data
        # ----------------------------------------------------

        try:
            cyclone_data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid cyclone data JSON."
            )

        required_fields = [
            "current_lat",
            "current_lon",
            "current_wind",
            "speed",
            "direction",
            "lat_3h_ago",
            "lon_3h_ago",
            "lat_6h_ago",
            "lon_6h_ago"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in cyclone_data
        ]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing_fields}"
            )

        # ----------------------------------------------------
        # Validate coordinates
        # ----------------------------------------------------

        current_lat = safe_float(cyclone_data["current_lat"])
        current_lon = safe_float(cyclone_data["current_lon"])
        direction = safe_float(cyclone_data["direction"])

        if not -90 <= current_lat <= 90:
            raise HTTPException(
                status_code=400,
                detail="Latitude must be between -90 and 90."
            )

        if not -180 <= current_lon <= 180:
            raise HTTPException(
                status_code=400,
                detail="Longitude must be between -180 and 180."
            )

        if not 0 <= direction <= 360:
            raise HTTPException(
                status_code=400,
                detail="Direction must be between 0 and 360 degrees."
            )

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not image.content_type:
            raise HTTPException(
                status_code=400,
                detail="Image content type is missing."
            )

        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid image."
            )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        safe_filename = os.path.basename(image.filename or "satellite.jpg")

        image_path = UPLOAD_FOLDER / safe_filename

        file_bytes = await image.read()

        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        with open(image_path, "wb") as f:
            f.write(file_bytes)

        # ----------------------------------------------------
        # 1. Satellite intensity
        # ----------------------------------------------------

        satellite_result = predict_insat_intensity(
            str(image_path)
        )

        # ----------------------------------------------------
        # 2. +3h intensity
        # ----------------------------------------------------

        intensity_result = predict_ibtracs_intensity(
            cyclone_data
        )

        # ----------------------------------------------------
        # 3. 3h track
        # ----------------------------------------------------

        track_3h_result = predict_ibtracs_track(
            cyclone_data
        )

        # ----------------------------------------------------
        # 4. 3–24h history track
        # ----------------------------------------------------

        history_result = predict_history_track(
            cyclone_data
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        result = {
            "status": "success",

            "satellite_intensity": satellite_result,

            "intensity_forecast": intensity_result,

            "track_3h": track_3h_result,

            "track_forecast": history_result,

            "current_position": {
                "latitude": round(current_lat, 4),
                "longitude": round(current_lon, 4)
            },

            "input": {
                "current_lat": current_lat,
                "current_lon": current_lon,
                "current_wind": safe_float(
                    cyclone_data["current_wind"]
                ),
                "speed": safe_float(
                    cyclone_data["speed"]
                ),
                "direction": direction
            }
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:

        print("Prediction error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        # Delete uploaded image after prediction
        if image_path is not None:

            try:
                if image_path.exists():
                    image_path.unlink()
            except Exception:
                pass

        gc.collect()


# ============================================================
# UPLOAD TEST
# ============================================================

@app.post("/api/upload-test")
async def upload_test(
    image: UploadFile = File(...)
):

    try:

        filename = os.path.basename(
            image.filename or "test.jpg"
        )

        save_path = UPLOAD_FOLDER / filename

        contents = await image.read()

        with open(save_path, "wb") as f:
            f.write(contents)

        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "filename": filename,
            "size_bytes": len(contents)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )