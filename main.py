from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import Optional
from sklearn.preprocessing import LabelEncoder

# ============================================
# ✅ LOAD ALL MODELS & SCALERS
# ============================================

model_full = joblib.load("model_full_38_cnn_lstm.pkl")
model_mi = joblib.load("model_mi_10_cnn_lstm.pkl")
model_rf = joblib.load("model_rf_10_cnn_lstm.pkl")

scaler_full = joblib.load("scaler_full_38.pkl")
scaler_mi = joblib.load("scaler_mi_10.pkl")
scaler_rf = joblib.load("scaler_rf_10.pkl")

print("✅ All models & scalers loaded!")

# ============================================
# ✅ FITUR UNTUK MASING-MASING MODEL
# ============================================

# 39 Fitur untuk Model Full
FULL_FEATURES = [
    'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
    'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
    'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
    'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
    'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory',
    'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'
]

# 10 Fitur untuk Model MI
MI_FEATURES = [
    'PhysicalActivities', 'HadAngina', 'RemovedTeeth',
    'AlcoholDrinkers', 'FluVaxLast12', 'ChestScan',
    'Sex', 'GeneralHealth', 'RaceEthnicityCategory', 'LastCheckupTime'
]

# 10 Fitur untuk Model RF
RF_FEATURES = [
    'HadAngina', 'BMI', 'State', 'WeightInKilograms', 'HeightInMeters',
    'AgeCategory', 'SleepHours', 'PhysicalHealthDays', 'TetanusLast10Tdap', 'GeneralHealth'
]

# ============================================
# ✅ FASTAPI APP
# ============================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://chd-main.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "CHD Backend is running", "models": ["full (39)", "mi (10)", "rf (10)"]}

# ============================================
# ✅ INPUT DATA MODEL
# ============================================

class PredictionInput(BaseModel):
    State: Optional[float] = 0
    Sex: Optional[float] = 0
    GeneralHealth: Optional[float] = 0
    PhysicalHealthDays: Optional[float] = 0
    MentalHealthDays: Optional[float] = 0
    LastCheckupTime: Optional[float] = 0
    PhysicalActivities: Optional[float] = 0
    SleepHours: Optional[float] = 0
    RemovedTeeth: Optional[float] = 0
    HadAngina: Optional[float] = 0
    HadStroke: Optional[float] = 0
    HadAsthma: Optional[float] = 0
    HadSkinCancer: Optional[float] = 0
    HadCOPD: Optional[float] = 0
    HadDepressiveDisorder: Optional[float] = 0
    HadKidneyDisease: Optional[float] = 0
    HadArthritis: Optional[float] = 0
    HadDiabetes: Optional[float] = 0
    DeafOrHardOfHearing: Optional[float] = 0
    BlindOrVisionDifficulty: Optional[float] = 0
    DifficultyConcentrating: Optional[float] = 0
    DifficultyWalking: Optional[float] = 0
    DifficultyDressingBathing: Optional[float] = 0
    DifficultyErrands: Optional[float] = 0
    SmokerStatus: Optional[float] = 0
    ECigaretteUsage: Optional[float] = 0
    ChestScan: Optional[float] = 0
    RaceEthnicityCategory: Optional[float] = 0
    AgeCategory: Optional[float] = 0
    HeightInMeters: Optional[float] = 0
    WeightInKilograms: Optional[float] = 0
    BMI: Optional[float] = 0
    AlcoholDrinkers: Optional[float] = 0
    HIVTesting: Optional[float] = 0
    FluVaxLast12: Optional[float] = 0
    PneumoVaxEver: Optional[float] = 0
    TetanusLast10Tdap: Optional[float] = 0
    HighRiskLastYear: Optional[float] = 0
    CovidPos: Optional[float] = 0

# ============================================
# ✅ HELPER FUNCTION
# ============================================

def preprocess(data: PredictionInput, features: list, scaler):
    """Convert input data to numpy array, scale, and reshape for CNN-LSTM"""
    input_list = [float(getattr(data, f)) for f in features]
    input_array = np.array([input_list])
    input_scaled = scaler.transform(input_array)
    input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)
    return input_reshaped

# ============================================
# ✅ ENDPOINTS
# ============================================

@app.post("/predict/full")
def predict_full(data: PredictionInput):
    try:
        input_reshaped = preprocess(data, FULL_FEATURES, scaler_full)
        proba = model_full.predict(input_reshaped)
        prediction = int(proba[0][0] >= 0.5)
        return {
            "model": "CNN-LSTM (38 fitur)",
            "prediction": prediction,
            "probability": float(proba[0][0])
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/mi")
def predict_mi(data: PredictionInput):
    try:
        input_reshaped = preprocess(data, MI_FEATURES, scaler_mi)
        proba = model_mi.predict(input_reshaped)
        prediction = int(proba[0][0] >= 0.5)
        return {
            "model": "CNN-LSTM + MI (10 fitur)",
            "prediction": prediction,
            "probability": float(proba[0][0])
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/rf")
def predict_rf(data: PredictionInput):
    try:
        input_reshaped = preprocess(data, RF_FEATURES, scaler_rf)
        proba = model_rf.predict(input_reshaped)
        prediction = int(proba[0][0] >= 0.5)
        return {
            "model": "CNN-LSTM + RF (10 fitur)",
            "prediction": prediction,
            "probability": float(proba[0][0])
        }
    except Exception as e:
        return {"error": str(e)}