
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import pandas as pd
import numpy as np
import joblib
import httpx


# ==================================================
# 1. CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="SmartCrop API",
    description="ML-Based Crop Recommendation System",
    version="1.1"
)


# ==================================================
# 2. LOAD SAVED MACHINE LEARNING FILES
# ==================================================

model = joblib.load("saved_models/crop_model.pkl")

label_encoder = joblib.load(
    "saved_models/label_encoder.pkl"
)

feature_columns = joblib.load(
    "saved_models/feature_columns.pkl"
)

fertilizer_guidance = joblib.load(
    "saved_models/fertilizer_guidance.pkl"
)

print("SmartCrop models loaded successfully!")


# ==================================================
# 3. INPUT DATA STRUCTURES
# ==================================================

class CropInput(BaseModel):

    country: str
    state: str
    district: str

    N: int
    P: int
    K: int

    temperature: float = Field(allow_inf_nan=False)
    humidity: float = Field(allow_inf_nan=False)
    ph: float = Field(allow_inf_nan=False)
    rainfall: float = Field(allow_inf_nan=False)


class AutoCropInput(BaseModel):

    place: str

    N: int
    P: int
    K: int

    ph: float = Field(allow_inf_nan=False)

    # This is the rainfall value used by the trained ML model.
    # It is kept separate from automatically fetched 30-day rainfall.
    rainfall: float = Field(allow_inf_nan=False)


# ==================================================
# 4. DATASET INPUT LIMITS
# ==================================================

DATASET_LIMITS = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),

    "temperature": (8.825675, 43.675493),
    "humidity": (14.258040, 99.981876),
    "ph": (3.504752, 9.935091),
    "rainfall": (20.211267, 298.56011)
}


# ==================================================
# 5. DATASET STATISTICS
# ==================================================

DATASET_STATS = {
    "N": {"mean": 50.55, "std": 36.92},
    "P": {"mean": 53.36, "std": 32.99},
    "K": {"mean": 48.15, "std": 50.65},

    "temperature": {"mean": 25.62, "std": 5.06},
    "humidity": {"mean": 71.48, "std": 22.26},
    "ph": {"mean": 6.47, "std": 0.77},
    "rainfall": {"mean": 103.46, "std": 54.96}
}


# ==================================================
# 6. LOCATION VALIDATION

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import httpx

app = FastAPI(
    title="SmartCrop API",
    description="ML-Based Crop Recommendation System",
    version="2.0"
)

# 1. LOAD TRAINED MODEL FILES
model = joblib.load("saved_models/crop_model.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")
feature_columns = joblib.load("saved_models/feature_columns.pkl")

print("SmartCrop model loaded successfully!")

# 2. SOIL SUITABILITY REFERENCE
# Soil type is advisory information, NOT an ML feature.

soil_suitability = {
    "rice": ["Clay soil", "Clay loam", "Alluvial soil"],
    "maize": ["Well-drained loamy soil", "Sandy loam", "Silt loam"],
    "chickpea": ["Clay loam", "Loam", "Sandy loam"],
    "kidneybeans": ["Well-drained loam", "Sandy loam", "Silt loam"],
    "pigeonpeas": ["Well-drained black soil", "Sandy loam", "Clay loam"],
    "mothbeans": ["Sandy soil", "Sandy loam", "Loamy sand"],
    "mungbean": ["Well-drained loam", "Sandy loam", "Silt loam"],
    "blackgram": ["Loam", "Sandy loam", "Clay loam"],
    "lentil": ["Clay loam", "Loam", "Silt loam"],
    "pomegranate": ["Well-drained loam", "Sandy loam", "Sandy soil"],
    "banana": ["Deep loamy soil", "Alluvial soil", "Clay loam with good drainage"],
    "mango": ["Deep loam", "Sandy loam", "Alluvial soil"],
    "grapes": ["Well-drained sandy loam", "Loam", "Gravelly loam"],
    "watermelon": ["Sandy loam", "Loamy sand", "Well-drained alluvial soil"],
    "muskmelon": ["Sandy loam", "Loamy sand", "Alluvial loam"],
    "apple": ["Well-drained loam", "Sandy loam", "Silt loam"],
    "orange": ["Deep well-drained loam", "Sandy loam", "Alluvial loam"],
    "papaya": ["Well-drained loam", "Sandy loam", "Alluvial soil"],
    "coconut": ["Sandy loam", "Alluvial soil", "Lateritic soil"],
    "cotton": ["Black cotton soil", "Alluvial loam", "Well-drained clay loam"],
    "jute": ["Alluvial soil", "Loam", "Clay loam"],
    "coffee": ["Deep, well-drained loamy soil", "Lateritic soil", "Clay loam with good drainage"]
}

# 3. GENERAL NPK FERTILIZER GUIDANCE

general_fertilizer_guidance = {
    "Nitrogen (N)": ["Urea"],
    "Phosphorus (P)": ["DAP", "SSP"],
    "Potassium (K)": ["MOP"]
}

fertilizer_note = (
    "Actual fertilizer requirements depend on soil-test results, "
    "crop variety, growth stage, soil conditions and expected yield."
)

# 4. INPUT STRUCTURES

class CropInput(BaseModel):
    country: str
    state: str
    district: str
    N: int
    P: int
    K: int
    temperature: float = Field(allow_inf_nan=False)
    humidity: float = Field(allow_inf_nan=False)
    ph: float = Field(allow_inf_nan=False)
    rainfall: float = Field(allow_inf_nan=False)


class AutoCropInput(BaseModel):
    place: str
    N: int
    P: int
    K: int
    ph: float = Field(allow_inf_nan=False)
    rainfall: float = Field(allow_inf_nan=False)


# 5. DATASET LIMITS

DATASET_LIMITS = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (8.825675, 43.675493),
    "humidity": (14.258040, 99.981876),
    "ph": (3.504752, 9.935091),
    "rainfall": (20.211267, 298.56011)
}

# 6. DATASET STATISTICS

DATASET_STATS = {
    "N": {"mean": 50.55, "std": 36.92},
    "P": {"mean": 53.36, "std": 32.99},
    "K": {"mean": 48.15, "std": 50.65},
    "temperature": {"mean": 25.62, "std": 5.06},
    "humidity": {"mean": 71.48, "std": 22.26},
    "ph": {"mean": 6.47, "std": 0.77},
    "rainfall": {"mean": 103.46, "std": 54.96}
}


# 7. VALIDATION

def validate_location(country, state, district):

    if country.strip().lower() != "india":
        raise HTTPException(
            status_code=400,
            detail="SmartCrop currently supports Indian agricultural regions only."
        )

    if not state.strip() or not district.strip():
        raise HTTPException(
            status_code=422,
            detail="Please provide a valid state and district."
        )


def validate_crop_input(data):

    values = {
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }

    errors = []

    for feature, value in values.items():

        minimum, maximum = DATASET_LIMITS[feature]

        if value < minimum or value > maximum:
            errors.append(
                f"{feature} must be between {minimum} and {maximum}"
            )

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input values are outside training dataset ranges.",
                "errors": errors
            }
        )


def check_distribution(data):

    values = {
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }

    warnings = []

    for feature, value in values.items():

        mean = DATASET_STATS[feature]["mean"]
        std = DATASET_STATS[feature]["std"]

        if abs(value - mean) > 2 * std:
            warnings.append(
                f"{feature} is unusually far from the training dataset mean."
            )

    return warnings


# 8. TOP 3 CROP PREDICTIONS + SOIL SUITABILITY

def generate_recommendations(input_values):

    input_df = pd.DataFrame(
        [[input_values[column] for column in feature_columns]],
        columns=feature_columns
    )

    probabilities = model.predict_proba(input_df)[0]

    top3_indices = np.argsort(probabilities)[-3:][::-1]

    recommendations = []

    for index in top3_indices:

        crop = label_encoder.inverse_transform([index])[0]

        suitable_soils = soil_suitability.get(
            crop.lower(),
            ["Soil suitability information unavailable"]
        )

        recommendations.append({
            "priority": len(recommendations) + 1,
            "crop": crop,
            "suitable_soil_types": suitable_soils[:3],
            "soil_note": (
                "General crop-soil reference only. "
                "Confirm actual field soil through soil testing."
            )
        })

    return recommendations


# 9. GENERAL FERTILIZER RESPONSE

def fertilizer_response():

    return {
        "general_npk_sources": general_fertilizer_guidance,
        "important_note": fertilizer_note
    }


# 10. WEATHER RETRIEVAL

async def fetch_weather(latitude, longitude):

    async with httpx.AsyncClient(timeout=20) as client:

        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum",
                "timezone": "auto",
                "forecast_days": 1
            }
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Unable to retrieve weather information."
            )

        weather_data = response.json()

    return {
        "temperature": weather_data["current"]["temperature_2m"],
        "humidity": weather_data["current"]["relative_humidity_2m"],
        "rainfall_forecast": weather_data["daily"]["precipitation_sum"][0],
        "units": {
            "temperature": "°C",
            "humidity": "%",
            "rainfall_forecast": "mm"
        },
        "note": (
            "Forecast rainfall is shown separately. "
            "The model uses the manually entered rainfall value."
        )
    }


# 11. HOME ENDPOINT

@app.get("/")
def home():

    return {
        "message": "Welcome to SmartCrop API",
        "status": "API is running"
    }


# 12. MANUAL CROP PREDICTION

@app.post("/predict")
def predict_crop(data: CropInput):

    validate_location(data.country, data.state, data.district)
    validate_crop_input(data)

    warnings = check_distribution(data)

    input_values = {
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }

    recommendations = generate_recommendations(input_values)

    return {
        "status": "success",

        "location": {
            "country": data.country,
            "state": data.state,
            "district": data.district
        },

        "recommendations": recommendations,

        "general_fertilizer_guidance": fertilizer_response(),

        "warnings": warnings
    }


# 13. AUTOMATIC GPS LOCATION

@app.get("/location-auto")
async def get_location(latitude: float, longitude: float):

    async with httpx.AsyncClient(timeout=20) as client:

        response = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "addressdetails": 1
            },
            headers={"User-Agent": "SmartCrop/2.0"}
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Location service is unavailable."
            )

        result = response.json()

    address = result.get("address", {})

    if address.get("country_code", "").lower() != "in":
        raise HTTPException(
            status_code=400,
            detail="SmartCrop currently supports Indian agricultural regions only."
        )

    return {
        "status": "success",
        "location": {
            "country": address.get("country", ""),
            "state": address.get("state", ""),
            "district": (
                address.get("state_district")
                or address.get("district")
                or address.get("county")
                or ""
            )
        },
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude
        }
    }


# 14. WEATHER USING GPS

@app.get("/weather-auto")
async def get_weather_auto(latitude: float, longitude: float):

    weather = await fetch_weather(latitude, longitude)

    return {
        "status": "success",
        "weather": weather
    }


# 15. WEATHER BY PLACE

@app.get("/weather-by-place")
async def weather_by_place(place: str):

    async with httpx.AsyncClient(timeout=20) as client:

        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": place,
                "format": "jsonv2",
                "addressdetails": 1,
                "countrycodes": "in",
                "limit": 1
            },
            headers={"User-Agent": "SmartCrop/2.0"}
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Location service is unavailable."
            )

        locations = response.json()

    if not locations:
        raise HTTPException(
            status_code=404,
            detail="Location not found."
        )

    location = locations[0]
    address = location.get("address", {})

    latitude = float(location["lat"])
    longitude = float(location["lon"])

    weather = await fetch_weather(latitude, longitude)

    return {
        "status": "success",
        "location": {
            "place": location.get("display_name", ""),
            "country": address.get("country", ""),
            "state": address.get("state", ""),
            "district": (
                address.get("state_district")
                or address.get("district")
                or address.get("county")
                or ""
            )
        },
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude
        },
        "weather": weather
    }


# 16. AUTOMATIC PLACE-BASED PREDICTION

@app.post("/predict-auto")
async def predict_auto(data: AutoCropInput):

    async with httpx.AsyncClient(timeout=20) as client:

        location_response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": data.place,
                "format": "jsonv2",
                "addressdetails": 1,
                "countrycodes": "in",
                "limit": 1
            },
            headers={"User-Agent": "SmartCrop/2.0"}
        )

        if location_response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Location service unavailable."
            )

        locations = location_response.json()

    if not locations:
        raise HTTPException(
            status_code=404,
            detail="Indian location not found."
        )

    location = locations[0]
    address = location.get("address", {})

    if address.get("country_code", "").lower() != "in":
        raise HTTPException(
            status_code=400,
            detail="SmartCrop supports Indian locations only."
        )

    latitude = float(location["lat"])
    longitude = float(location["lon"])

    weather = await fetch_weather(latitude, longitude)

    input_values = {
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "ph": data.ph,
        "rainfall": data.rainfall
    }

    class ModelInput(BaseModel):
        N: int
        P: int
        K: int
        temperature: float
        humidity: float
        ph: float
        rainfall: float

    validated_input = ModelInput(**input_values)

    validate_crop_input(validated_input)

    warnings = check_distribution(validated_input)

    recommendations = generate_recommendations(input_values)

    return {
        "status": "success",

        "location": {
            "country": address.get("country", ""),
            "state": address.get("state", ""),
            "district": (
                address.get("state_district")
                or address.get("district")
                or address.get("county")
                or ""
            )
        },

        "weather": weather,

        "recommendations": recommendations,

        "general_fertilizer_guidance": fertilizer_response(),

        "warnings": warnings
    }
