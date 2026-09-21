from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="NeuroVortex API",
    description="AI-based cyclone intensity, track and risk prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://neuro-vortex-frontend-phi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "backend",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

insat_model = tf.keras.models.load_model(
    os.path.join(
        MODEL_FOLDER,
        "best_insat3d_model.keras"
    )
)

track_model = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "cyclone_track_model.pkl"
    )
)

history_track_model = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "improved_history_track_model.pkl"
    )
)

intensity_model = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "cyclone_intensity_forecast_model.pkl"
    )
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "NeuroVortex API is running",
        "status": "success"
    }


# ============================================================
# API STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    return {
        "project": "NeuroVortex",
        "status": "ready",

        "models": {
            "insat3d": "ready",
            "ibtracs_intensity": "ready",
            "ibtracs_track": "ready",
            "history_track": "ready"
        }
    }


# ============================================================
# SAMPLE PREDICTION
# ============================================================

@app.get("/api/sample-prediction")
def sample_prediction():

    return {

        "satellite": {

            "intensity_kt": 57.12,

            "intensity_kmh": 105.79,

            "category": "Severe Cyclonic Storm",

            "risk": "MODERATE"
        },


        "intensity_forecast_3h": {

            "current_wind_kt": 73.0,

            "predicted_wind_kt": 70.58,

            "predicted_wind_kmh": 130.72,

            "category": "Very Severe Cyclonic Storm",

            "risk": "HIGH"
        },


        "track_3h": {

            "latitude": 10.8032,

            "longitude": 72.3205
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
        }
    }


# ============================================================
# INPUT MODEL
# ============================================================

class CycloneInput(BaseModel):

    current_lat: float

    current_lon: float

    current_wind: float

    speed: float

    direction: float

    lat_3h_ago: float

    lon_3h_ago: float

    lat_6h_ago: float

    lon_6h_ago: float


# ============================================================
# MAIN PREDICTION API
# ============================================================

@app.post("/api/predict")
async def predict_cyclone(

    data: str = Form(...),

    image: UploadFile = File(...)

):

    try:

        # ----------------------------------------------------
        # SAVE UPLOADED IMAGE
        # ----------------------------------------------------

        os.makedirs(
            UPLOAD_FOLDER,
            exist_ok=True
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            image.filename
        )

        with open(
            image_path,
            "wb"
        ) as buffer:

            buffer.write(
                await image.read()
            )


        # ----------------------------------------------------
        # VALIDATE INPUT DATA
        # ----------------------------------------------------

        data = CycloneInput.model_validate_json(
            data
        )


        # ----------------------------------------------------
        # SATELLITE INTENSITY
        # ----------------------------------------------------

        satellite_result = predict_insat_intensity_api(
            image_path
        )


        # ----------------------------------------------------
        # 3-HOUR INTENSITY FORECAST
        # ----------------------------------------------------

        intensity_result = predict_ibtracs_intensity_api(

            data.current_lat,

            data.current_lon,

            data.current_wind,

            data.speed,

            data.direction
        )


        # ----------------------------------------------------
        # 3-HOUR TRACK PREDICTION
        # ----------------------------------------------------

        track_result = predict_ibtracs_track_api(

            data.current_lat,

            data.current_lon,

            data.speed,

            data.direction
        )


        # ----------------------------------------------------
        # 3–24 HOUR TRACK FORECAST
        # ----------------------------------------------------

        history_result = predict_history_track_api(

            data.current_lat,

            data.current_lon,

            data.speed,

            data.direction,

            data.lat_3h_ago,

            data.lon_3h_ago,

            data.lat_6h_ago,

            data.lon_6h_ago
        )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "message": "Cyclone prediction generated successfully",

            "satellite_intensity": satellite_result,

            "intensity_forecast": intensity_result,

            "track_3h": track_result,

            "track_forecast": history_result
        }


    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# ============================================================
# UPLOAD TEST API
# ============================================================

@app.post("/api/upload-test")
async def upload_test(

    image: UploadFile = File(...)

):

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )


    image_path = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )


    with open(
        image_path,
        "wb"
    ) as buffer:

        buffer.write(
            await image.read()
        )


    return {

        "status": "success",

        "message": "Satellite image uploaded successfully",

        "filename": image.filename

    }


# ============================================================
# CYCLONE INTENSITY CLASSIFICATION
# ============================================================

def classify_cyclone_intensity(knots):

    if knots < 17:

        return "Low Pressure Area"

    elif knots < 28:

        return "Depression"

    elif knots < 34:

        return "Deep Depression"

    elif knots < 48:

        return "Cyclonic Storm"

    elif knots < 64:

        return "Severe Cyclonic Storm"

    elif knots < 90:

        return "Very Severe Cyclonic Storm"

    elif knots < 120:

        return "Extremely Severe Cyclonic Storm"

    else:

        return "Super Cyclonic Storm"


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_cyclone_risk(knots):

    if knots < 34:

        return "LOW"

    elif knots < 64:

        return "MODERATE"

    elif knots < 90:

        return "HIGH"

    else:

        return "VERY HIGH"


# ============================================================
# INSAT-3D INTENSITY PREDICTION
# ============================================================

def predict_insat_intensity_api(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")


    img = img.resize(
        (224, 224)
    )


    img = np.array(
        img
    ).astype("float32")


    img = np.expand_dims(
        img,
        axis=0
    )


    prediction_norm = float(
        insat_model.predict(
            img,
            verbose=0
        )[0][0]
    )


    # Same normalization used during training

    label_min = 25.0

    label_max = 128.0


    prediction_knots = (

        prediction_norm *
        (label_max - label_min)

        + label_min

    )


    return {

        "intensity_kt": round(
            prediction_knots,
            2
        ),

        "intensity_kmh": round(
            prediction_knots * 1.852,
            2
        ),

        "category": classify_cyclone_intensity(
            prediction_knots
        ),

        "risk": classify_cyclone_risk(
            prediction_knots
        )

    }


# ============================================================
# IBTRACS INTENSITY FORECAST
# ============================================================

def predict_ibtracs_intensity_api(

    current_lat,

    current_lon,

    current_wind,

    speed,

    direction

):

    direction_rad = np.deg2rad(
        direction
    )


    X = np.array([

        [

            current_lat,

            current_lon,

            current_wind,

            speed,

            np.sin(direction_rad),

            np.cos(direction_rad)

        ]

    ])


    predicted_wind = float(

        intensity_model.predict(
            X
        )[0]

    )


    return {

        "current_wind_kt": round(
            current_wind,
            2
        ),

        "predicted_wind_kt": round(
            predicted_wind,
            2
        ),

        "predicted_wind_kmh": round(
            predicted_wind * 1.852,
            2
        ),

        "category": classify_cyclone_intensity(
            predicted_wind
        ),

        "risk": classify_cyclone_risk(
            predicted_wind
        )

    }


# ============================================================
# IBTRACS TRACK PREDICTION
# ============================================================

def predict_ibtracs_track_api(

    current_lat,

    current_lon,

    speed,

    direction

):

    direction_rad = np.deg2rad(
        direction
    )


    X = np.array([

        [

            current_lat,

            current_lon,

            speed,

            np.sin(direction_rad),

            np.cos(direction_rad)

        ]

    ])


    predicted_position = track_model.predict(
        X
    )[0]


    return {

        "latitude": round(
            float(predicted_position[0]),
            4
        ),

        "longitude": round(
            float(predicted_position[1]),
            4
        )

    }


# ============================================================
# HISTORY-BASED TRACK PREDICTION
# ============================================================

def predict_history_track_api(

    current_lat,

    current_lon,

    speed,

    direction,

    lat_3h_ago,

    lon_3h_ago,

    lat_6h_ago,

    lon_6h_ago

):

    direction_rad = np.deg2rad(
        direction
    )


    # Movement features

    lat_change_3h = (
        current_lat -
        lat_3h_ago
    )

    lon_change_3h = (
        current_lon -
        lon_3h_ago
    )


    lat_change_6h = (
        current_lat -
        lat_6h_ago
    )


    lon_change_6h = (
        current_lon -
        lon_6h_ago
    )


    X = np.array([

        [

            lat_6h_ago,

            lon_6h_ago,

            lat_3h_ago,

            lon_3h_ago,

            current_lat,

            current_lon,

            speed,

            np.sin(direction_rad),

            np.cos(direction_rad),

            lat_change_3h,

            lon_change_3h,

            lat_change_6h,

            lon_change_6h

        ]

    ])


    prediction = history_track_model.predict(
        X
    )[0]


    return {

        "3h": {

            "latitude": round(
                float(prediction[0]),
                4
            ),

            "longitude": round(
                float(prediction[1]),
                4
            )

        },

        "6h": {

            "latitude": round(
                float(prediction[2]),
                4
            ),

            "longitude": round(
                float(prediction[3]),
                4
            )

        },

        "12h": {

            "latitude": round(
                float(prediction[4]),
                4
            ),

            "longitude": round(
                float(prediction[5]),
                4
            )

        },

        "18h": {

            "latitude": round(
                float(prediction[6]),
                4
            ),

            "longitude": round(
                float(prediction[7]),
                4
            )

        },

        "24h": {

            "latitude": round(
                float(prediction[8]),
                4
            ),

            "longitude": round(
                float(prediction[9]),
                4
            )

        }

    }