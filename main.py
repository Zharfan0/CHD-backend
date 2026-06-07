from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any
import numpy as np
import joblib
import tensorflow as tf

# ============================================================
# LOAD ARTEFAK — baca dari metadata, TIDAK hardcode fitur
# ============================================================

# --- Model (format .keras) ---
model_rf   = tf.keras.models.load_model("model_rf_cnn_lstm.keras")
model_mi   = tf.keras.models.load_model("model_mi_cnn_lstm.keras")
model_full = tf.keras.models.load_model("model_full_cnn_lstm.keras")

# --- Scaler ---
scaler_rf   = joblib.load("scaler_rf.pkl")
scaler_mi   = joblib.load("scaler_mi.pkl")
scaler_full = joblib.load("scaler_full.pkl")

# --- Encoder per kolom ---
encoders_rf   = joblib.load("encoders_rf.pkl")
encoders_mi   = joblib.load("encoders_mi.pkl")
encoders_full = joblib.load("encoders_full.pkl")

# --- Metadata (berisi selected_features & encoder_types) ---
metadata_rf   = joblib.load("metadata_rf.pkl")
metadata_mi   = joblib.load("metadata_mi.pkl")
metadata_full = joblib.load("metadata_full.pkl")

# Ambil urutan fitur langsung dari metadata — TIDAK hardcode
RF_FEATURES   = metadata_rf["selected_features"]    # 34 fitur
MI_FEATURES   = metadata_mi["selected_features"]    # 35 fitur
FULL_FEATURES = metadata_full["selected_features"]  # 37 fitur

# Tipe encoder per kolom (hanya ada di MI)
MI_ENCODER_TYPES = metadata_mi.get("encoder_types", {})

print("✅ Semua model, scaler, encoder, dan metadata berhasil di-load!")
print(f"   RF   : {len(RF_FEATURES)} fitur")
print(f"   MI   : {len(MI_FEATURES)} fitur")
print(f"   Full : {len(FULL_FEATURES)} fitur")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="CHD Prediction API", version="2.0")

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
    return {
        "status" : "CHD Backend v2.0 is running",
        "target" : "HadAngina (Coronary Heart Disease)",
        "models" : {
            "rf"   : f"CNN-LSTM + RF Feature Importance ({len(RF_FEATURES)} fitur)",
            "mi"   : f"CNN-LSTM + Mutual Information ({len(MI_FEATURES)} fitur)",
            "full" : f"CNN-LSTM Full ({len(FULL_FEATURES)} fitur)",
        }
    }


@app.get("/features")
def get_features():
    """Endpoint untuk frontend mengetahui fitur yang dibutuhkan tiap model."""
    return {
        "rf"   : RF_FEATURES,
        "mi"   : MI_FEATURES,
        "full" : FULL_FEATURES,
    }


# ============================================================
# INPUT DATA MODEL
# Semua fitur yang mungkin dibutuhkan oleh salah satu model
# ============================================================

class PredictionInput(BaseModel):
    Sex:                    Optional[Any] = None
    GeneralHealth:          Optional[Any] = None
    PhysicalHealthDays:     Optional[Any] = None
    MentalHealthDays:       Optional[Any] = None
    LastCheckupTime:        Optional[Any] = None
    PhysicalActivities:     Optional[Any] = None
    SleepHours:             Optional[Any] = None
    RemovedTeeth:           Optional[Any] = None
    HadStroke:              Optional[Any] = None
    HadAsthma:              Optional[Any] = None
    HadSkinCancer:          Optional[Any] = None
    HadCOPD:                Optional[Any] = None
    HadDepressiveDisorder:  Optional[Any] = None
    HadKidneyDisease:       Optional[Any] = None
    HadArthritis:           Optional[Any] = None
    HadDiabetes:            Optional[Any] = None
    DeafOrHardOfHearing:    Optional[Any] = None
    BlindOrVisionDifficulty:Optional[Any] = None
    DifficultyConcentrating:Optional[Any] = None
    DifficultyWalking:      Optional[Any] = None
    DifficultyDressingBathing:Optional[Any] = None
    DifficultyErrands:      Optional[Any] = None
    SmokerStatus:           Optional[Any] = None
    ECigaretteUsage:        Optional[Any] = None
    ChestScan:              Optional[Any] = None
    RaceEthnicityCategory:  Optional[Any] = None
    AgeCategory:            Optional[Any] = None
    HeightInMeters:         Optional[Any] = None
    WeightInKilograms:      Optional[Any] = None
    BMI:                    Optional[Any] = None
    AlcoholDrinkers:        Optional[Any] = None
    HIVTesting:             Optional[Any] = None
    FluVaxLast12:           Optional[Any] = None
    PneumoVaxEver:          Optional[Any] = None
    TetanusLast10Tdap:      Optional[Any] = None
    HighRiskLastYear:       Optional[Any] = None
    CovidPos:               Optional[Any] = None


# ============================================================
# HELPER: ENCODE SATU NILAI
# ============================================================

def encode_value(col: str, value: Any, encoders: dict,
                 encoder_types: dict = None) -> float:
    """
    Encode satu nilai input sesuai encoder yang dipakai saat training.

    - Jika nilai sudah numerik → langsung pakai
    - Jika ada encoder untuk kolom ini → transform
    - Kalau nilai tidak dikenal encoder → kembalikan 0 (default aman)
    """
    if value is None:
        return 0.0

    # Sudah numerik, tidak perlu encode
    if isinstance(value, (int, float)):
        return float(value)

    # Nilai berupa string → perlu encode
    str_val = str(value)

    if col not in encoders:
        # Kolom numerik asli, coba konversi langsung
        try:
            return float(str_val)
        except ValueError:
            return 0.0

    enc = encoders[col]
    enc_type = (encoder_types or {}).get(col, "LabelEncoder")

    try:
        if enc_type == "OrdinalEncoder":
            # OrdinalEncoder expects 2D array
            result = enc.transform([[str_val]])
            return float(result[0][0])
        else:
            # LabelEncoder expects 1D
            result = enc.transform([str_val])
            return float(result[0])
    except (ValueError, KeyError):
        # Nilai tidak dikenal encoder → kembalikan -1 (unknown_value OrdinalEncoder)
        # atau 0 untuk LabelEncoder
        return -1.0 if enc_type == "OrdinalEncoder" else 0.0


# ============================================================
# HELPER: PREPROCESS LENGKAP
# ============================================================

def preprocess(data: PredictionInput,
               features: list,
               scaler,
               encoders: dict,
               encoder_types: dict = None) -> np.ndarray:
    """
    1. Ambil nilai tiap fitur dari input sesuai urutan `features`
    2. Encode nilai string → integer sesuai encoder training
    3. Scale dengan scaler training
    4. Reshape ke (1, n_features, 1) untuk CNN-LSTM
    """
    input_list = []
    for col in features:
        raw = getattr(data, col, None)
        encoded = encode_value(col, raw, encoders, encoder_types)
        input_list.append(encoded)

    input_array  = np.array([input_list], dtype=np.float32)
    input_scaled = scaler.transform(input_array)
    input_3d     = input_scaled.reshape(1, input_scaled.shape[1], 1)
    return input_3d


# ============================================================
# ENDPOINTS PREDIKSI
# ============================================================

@app.post("/predict/rf")
def predict_rf(data: PredictionInput):
    """Prediksi dengan CNN-LSTM + RF Feature Importance (34 fitur)."""
    try:
        X = preprocess(data, RF_FEATURES, scaler_rf, encoders_rf)
        proba = float(model_rf.predict(X, verbose=0)[0][0])
        return {
            "model"      : f"CNN-LSTM + RF ({len(RF_FEATURES)} fitur)",
            "prediction" : int(proba >= 0.5),
            "probability": round(proba, 4),
            "features_used": RF_FEATURES,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/mi")
def predict_mi(data: PredictionInput):
    """Prediksi dengan CNN-LSTM + Mutual Information (35 fitur)."""
    try:
        X = preprocess(data, MI_FEATURES, scaler_mi, encoders_mi,
                       encoder_types=MI_ENCODER_TYPES)
        proba = float(model_mi.predict(X, verbose=0)[0][0])
        return {
            "model"      : f"CNN-LSTM + MI ({len(MI_FEATURES)} fitur)",
            "prediction" : int(proba >= 0.5),
            "probability": round(proba, 4),
            "features_used": MI_FEATURES,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/full")
def predict_full(data: PredictionInput):
    """Prediksi dengan CNN-LSTM Full (37 fitur, tanpa seleksi)."""
    try:
        X = preprocess(data, FULL_FEATURES, scaler_full, encoders_full)
        proba = float(model_full.predict(X, verbose=0)[0][0])
        return {
            "model"      : f"CNN-LSTM Full ({len(FULL_FEATURES)} fitur)",
            "prediction" : int(proba >= 0.5),
            "probability": round(proba, 4),
            "features_used": FULL_FEATURES,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ENDPOINT PREDIKSI SEMUA MODEL SEKALIGUS
# ============================================================

@app.post("/predict/all")
def predict_all(data: PredictionInput):
    """Jalankan ketiga model sekaligus dan kembalikan perbandingan."""
    results = {}
    for name, feats, scaler, encoders, enc_types, model in [
        ("rf",   RF_FEATURES,   scaler_rf,   encoders_rf,   None,              model_rf),
        ("mi",   MI_FEATURES,   scaler_mi,   encoders_mi,   MI_ENCODER_TYPES,  model_mi),
        ("full", FULL_FEATURES, scaler_full, encoders_full, None,              model_full),
    ]:
        try:
            X     = preprocess(data, feats, scaler, encoders, enc_types)
            proba = float(model.predict(X, verbose=0)[0][0])
            results[name] = {
                "prediction" : int(proba >= 0.5),
                "probability": round(proba, 4),
                "n_features" : len(feats),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "target" : "HadAngina (CHD)",
        "results": results,
    }
