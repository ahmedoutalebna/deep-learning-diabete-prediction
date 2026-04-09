from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from typing import List

# ============================================
# 1. LOAD MODEL AND SCALER
# ============================================

# Load the trained model
# IMPORTANT: This must match EXACTLY the model architecture used during training
model = nn.Sequential(
    nn.Linear(21, 128),  # Input: 21 features
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()  # Output activation (must match training model!)
)

# Load saved weights
model.load_state_dict(torch.load('diabetes_model.pt'))
model.eval()  # Set to evaluation mode

# Load scaler
scaler = joblib.load('scaler.pkl')

# ============================================
# 2. CREATE FASTAPI APP
# ============================================

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes using a neural network",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 3. DEFINE REQUEST/RESPONSE MODELS
# ============================================

class PatientData(BaseModel):
    """
    Input model for patient health indicators
    
    Features (21 total - normalized between 0 and 1):
    0. HighBP - High Blood Pressure
    1. HighChol - High Cholesterol
    2. CholCheck - Cholesterol Check
    3. BMI - Body Mass Index
    4. Smoker - Smoking Status
    5. Stroke - History of Stroke
    6. Diabetes - Diabetes Status
    7. PhysActivity - Physical Activity
    8. Fruits - Fruit Consumption
    9. Veggies - Vegetable Consumption
    10. HvyAlcoholConsump - Heavy Alcohol Consumption
    11. AnyHealthcare - Any Healthcare Coverage
    12. NoDocbcCost - No Doctor Due to Cost
    13. GenHlth - General Health
    14. MentHlth - Mental Health
    15. PhysHlth - Physical Health
    16. DiffWalk - Difficulty Walking
    17. Sex - Gender
    18. Age - Age Group
    19. Income - Income Level
    20. Education - Education Level
    """
    features: List[float]
    
    class Config:
        example = {
            "features": {
                "HighBP": 0.0,
                "HighChol": 0.0,
                "CholCheck": 1.0,
                "BMI": 0.5,
                "Smoker": 0.0,
                "Stroke": 0.0,
                "Diabetes": 0.0,
                "PhysActivity": 1.0,
                "Fruits": 1.0,
                "Veggies": 1.0,
                "HvyAlcoholConsump": 0.0,
                "AnyHealthcare": 1.0,
                "NoDocbcCost": 0.0,
                "GenHlth": 0.5,
                "MentHlth": 0.5,
                "PhysHlth": 0.5,
                "DiffWalk": 0.0,
                "Sex": 0.0,
                "Age": 0.5,
                "Income": 0.5,
                "Education": 0.5
        }}

class PredictionResponse(BaseModel):
    """Output model for predictions"""
    probability: float
    prediction: str
    threshold: float
    confidence: float

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[dict]
    count: int

# ============================================
# 4. DEFINE API ENDPOINTS
# ============================================

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Diabetes Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET - Interactive documentation)"
        }
    }

@app.get("/health")
def health_check():
    """Check if API is running"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_single(data: PatientData):
    """
    Predict diabetes for a single patient
    
    Input: 21 health indicators
    Output: Prediction (Diabetes/No Diabetes) with probability
    """
    try:
        # Validate input
        if len(data.features) != 21:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 21 features, got {len(data.features)}"
            )
        
        # Convert to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Scale features{"message":"Welcome to Diabetes Prediction API","version":"1.0.0","endpoints":{"predict":"/predict (POST)","predict_batch":"/predict_batch (POST)","health":"/health (GET)","docs":"/docs (GET - Interactive documentation)"}}
        features_scaled = scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = model(features_tensor)
            probability = output.item()  # Already has Sigmoid in model
        
        # Apply decision threshold
        threshold = 0.5
        prediction = "Diabetes" if probability >= threshold else "No Diabetes"
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            probability=round(probability, 4),
            prediction=prediction,
            threshold=threshold,
            confidence=round(confidence, 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(data_list: List[PatientData]):
    """
    Predict diabetes for multiple patients at once
    
    Input: List of patient data (max 100)
    Output: Batch predictions
    """
    try:
        if len(data_list) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 predictions per batch"
            )
        
        predictions = []
        
        for i, data in enumerate(data_list):
            if len(data.features) != 21:
                predictions.append({
                    "patient_id": i,
                    "error": f"Expected 21 features, got {len(data.features)}"
                })
                continue
            
            # Process patient
            features = np.array(data.features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                output = model(features_tensor)
                probability = torch.sigmoid(output).item()
            
            threshold = 0.5
            prediction = "Diabetes" if probability >= threshold else "No Diabetes"
            
            predictions.append({
                "patient_id": i,
                "probability": round(probability, 4),
                "prediction": prediction,
                "confidence": round(max(probability, 1 - probability), 4)
            })
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_threshold")
def predict_with_custom_threshold(data: PatientData, threshold: float = 0.5):
    """
    Predict with custom decision threshold
    Parameters:
    - features: List of 21 health indicators
    - threshold: Decision threshold (0.0 to 1.0)
    """
    try:
        if threshold < 0 or threshold > 1:
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0 and 1"
            )
        
        features = np.array(data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(features_tensor)
            probability = torch.sigmoid(output).item()
        
        prediction = "Diabetes" if probability >= threshold else "No Diabetes"
        
        return {
            "probability": round(probability, 4),
            "prediction": prediction,
            "threshold": threshold,
            "confidence": round(max(probability, 1 - probability), 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 5. RUN THE APP
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)