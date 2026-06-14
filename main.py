from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import tensorflow as tf
from typing import Optional

# ============================================
# LOAD MODELS
# ============================================

# Full CNN-LSTM (37 fitur)
model_full     = tf.keras.models.load_model("model_full_cnn_lstm.keras")
scaler_full    = joblib.load("scaler_full.pkl")   # ← sesuaikan nama filenya

# Two-Stage (CNN-LSTM MI k=15 + LR)
model_stage1   = tf.keras.models.load_model("model_stage1_cnn_lstm.keras")
model_stage2   = joblib.load("model_stage2_lr.pkl")
scaler_mi15    = joblib.load("scaler_mi15.pkl")

THR_STAGE1 = 0.30
THR_STAGE2 = 0.50

print("All models loaded!")

# ============================================
# FITUR
# ============================================

FULL_FEATURES = [
    'sex', 'generalhealth', 'physicalhealthdays', 'mentalhealthdays',
    'lastcheckuptime', 'physicalactivities', 'sleephours', 'removedteeth',
    'hadstroke', 'hadasthma', 'hadskincancer', 'hadcopd',
    'haddepressivedisorder', 'hadkidneydisease', 'hadarthritis', 'haddiabetes',
    'deaforhardofhearing', 'blindorvisiondifficulty', 'difficultyconcentrating',
    'difficultywalking', 'difficultydressingbathing', 'difficultyerrands',
    'smokerstatus', 'ecigaretteusage', 'chestscan', 'raceethnicitycategory',
    'agecategory', 'heightinmeters', 'weightinkilograms', 'bmi',
    'alcoholdrinkers', 'hivtesting', 'fluvaxlast12', 'pneumovaxever',
    'tetanuslast10tdap', 'highrisklastyear', 'covidpos'
]  # 37 fitur (HadAngina dikeluarkan karena target variable)

TWO_STAGE_FEATURES = [
    'physicalactivities', 'chestscan', 'alcoholdrinkers',
    'fluvaxlast12', 'lastcheckuptime', 'pneumovaxever',
    'generalhealth', 'sex', 'raceethnicitycategory',
    'agecategory', 'hadarthritis', 'removedteeth',
    'tetanuslast10tdap', 'hivtesting', 'difficultywalking'
]  # 15 fitur MI k=15

# ============================================
# APP
# ============================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# INPUT MODEL
# ============================================

class PredictionInput(BaseModel):
    sex: Optional[float] = 0
    generalhealth: Optional[float] = 0
    physicalhealthdays: Optional[float] = 0
    mentalhealthdays: Optional[float] = 0
    lastcheckuptime: Optional[float] = 0
    physicalactivities: Optional[float] = 0
    sleephours: Optional[float] = 0
    removedteeth: Optional[float] = 0
    hadstroke: Optional[float] = 0
    hadasthma: Optional[float] = 0
    hadskincancer: Optional[float] = 0
    hadcopd: Optional[float] = 0
    haddepressivedisorder: Optional[float] = 0
    hadkidneydisease: Optional[float] = 0
    hadarthritis: Optional[float] = 0
    haddiabetes: Optional[float] = 0
    deaforhardofhearing: Optional[float] = 0
    blindorvisiondifficulty: Optional[float] = 0
    difficultyconcentrating: Optional[float] = 0
    difficultywalking: Optional[float] = 0
    difficultydressingbathing: Optional[float] = 0
    difficultyerrands: Optional[float] = 0
    smokerstatus: Optional[float] = 0
    ecigaretteusage: Optional[float] = 0
    chestscan: Optional[float] = 0
    raceethnicitycategory: Optional[float] = 0
    agecategory: Optional[float] = 0
    heightinmeters: Optional[float] = 0
    weightinkilograms: Optional[float] = 0
    bmi: Optional[float] = 0
    alcoholdrinkers: Optional[float] = 0
    hivtesting: Optional[float] = 0
    fluvaxlast12: Optional[float] = 0
    pneumovaxever: Optional[float] = 0
    tetanuslast10tdap: Optional[float] = 0
    highrisklastyear: Optional[float] = 0
    covidpos: Optional[float] = 0

# ============================================
# HELPER
# ============================================

def preprocess_cnn(data: PredictionInput, features: list, scaler):
    input_arr    = np.array([[float(getattr(data, f)) for f in features]])
    input_scaled = scaler.transform(input_arr)
    return input_scaled, input_scaled.reshape(1, len(features), 1)

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "status": "MyHeartD Backend running",
        "models": {
            "full": f"{len(FULL_FEATURES)} fitur",
            "two-stage": f"{len(TWO_STAGE_FEATURES)} fitur (CNN-LSTM MI + LR)"
        }
    }

@app.get("/features/{model}")
def get_features(model: str):
    if model == "full":
        return {"model": "full", "features": FULL_FEATURES, "count": len(FULL_FEATURES)}
    elif model == "two-stage":
        return {"model": "two-stage", "features": TWO_STAGE_FEATURES, "count": len(TWO_STAGE_FEATURES)}
    return {"error": "Model tidak dikenal. Gunakan 'full' atau 'two-stage'"}

@app.post("/predict/full")
def predict_full(data: PredictionInput):
    try:
        _, input_3d = preprocess_cnn(data, FULL_FEATURES, scaler_full)
        proba       = float(model_full.predict(input_3d, verbose=0)[0][0])
        prediction  = int(proba >= 0.50)
        return {
            "model": "CNN-LSTM Full (37 fitur)",
            "prediction": prediction,
            "probability": proba
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/two-stage")
def predict_two_stage(data: PredictionInput):
    try:
        input_scaled, input_3d = preprocess_cnn(data, TWO_STAGE_FEATURES, scaler_mi15)

        # Stage 1 — CNN-LSTM
        s1_proba = float(model_stage1.predict(input_3d, verbose=0)[0][0])

        if s1_proba < THR_STAGE1:
            return {
                "model": "Two-Stage (early exit Stage 1)",
                "prediction": 0,
                "probability": round(1 - s1_proba, 4),
                "stage": 1,
                "stage1_proba": round(s1_proba, 4)
            }

        # Stage 2 — Logistic Regression
        s2_proba   = float(model_stage2.predict_proba(input_scaled)[0][1])
        prediction = int(s2_proba >= THR_STAGE2)

        return {
            "model": "Two-Stage (CNN-LSTM MI + LR)",
            "prediction": prediction,
            "probability": round(s2_proba, 4),
            "stage": 2,
            "stage1_proba": round(s1_proba, 4),
            "stage2_proba": round(s2_proba, 4)
        }
    except Exception as e:
        return {"error": str(e)}