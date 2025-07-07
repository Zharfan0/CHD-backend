from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import Optional


# ✅ Load model
model_rf = joblib.load("model_rf.pkl")
model_mi = joblib.load("model_mi_best.pkl")

# ✅ Inisialisasi FastAPI
app = FastAPI()

# ✅ Middleware CORS agar bisa diakses dari frontend React/Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Sesuaikan jika frontend deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Struktur data input dari frontend
class InputData(BaseModel):
    sex: Optional[float] = None
    generalhealth: Optional[float] = None
    physicalhealthdays: Optional[float] = None
    mentalhealthdays: Optional[float] = None
    sleephours: Optional[float] = None
    hadasthma: Optional[float] = None
    haddiabetes: Optional[float] = None
    raceethnicitycategory: Optional[float] = None
    agecategory: Optional[float] = None
    bmi: Optional[float] = None
    physicalactivities: Optional[float] = None
    removedteeth: Optional[float] = None
    alcoholdrinkers: Optional[float] = None
    fluvaxlast12: Optional[float] = None
    chestscan: Optional[float] = None
    lastcheckuptime: Optional[float] = None

# ✅ Preprocessing khusus untuk Random Forest
def preprocess_input_rf(data: InputData):
    fields = [
        "sex",
        "generalhealth",
        "physicalhealthdays",
        "mentalhealthdays",
        "sleephours",
        "hadasthma",
        "haddiabetes",
        "raceethnicitycategory",
        "agecategory",
        "bmi"
    ]
    return np.array([[float(getattr(data, f)) for f in fields]])

# ✅ Preprocessing khusus untuk MI + Random Forest
def preprocess_input_mi(data: InputData):
    fields = [
        "physicalactivities",
        "hadasthma",
        "removedteeth",
        "alcoholdrinkers",
        "fluvaxlast12",
        "chestscan",
        "sex",
        "generalhealth",
        "raceethnicitycategory",
        "lastcheckuptime"
    ]
    return np.array([[float(getattr(data, f)) for f in fields]])

@app.post("/predict/rf")
def predict_rf(data: InputData):
    try:
        input_array = preprocess_input_rf(data)
        prediction = model_rf.predict(input_array)

        # Ambil probabilitas dan ubah ke float biasa
        probability = float(prediction[0][0])

        # Konversi ke kelas 0/1
        kelas = int(probability >= 0.5)

        return {
            "prediction": kelas,
            "probability": probability
        }
    except Exception as e:
        return {"error": str(e)}



@app.post("/predict/mi")
def predict_mi(data: InputData):
    try:
        input_array = preprocess_input_mi(data)
        print("📥 Input array dari frontend:", input_array)

        # Keras model langsung pakai predict untuk probabilitas
        prediction_proba = model_mi.predict(input_array)
        probability = float(prediction_proba[0][0]) if prediction_proba.ndim == 2 else float(prediction_proba[0])

        kelas = int(probability >= 0.5)
        print("📈 Probabilitas hasil prediksi:", probability)

        return {
            "prediction": kelas,
            "probability": probability
        }
    except Exception as e:
        return {"error": str(e)}



