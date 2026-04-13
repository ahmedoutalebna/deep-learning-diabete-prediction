from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import PredictionResponse,  PatientData, BatchPredictionResponse
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from typing import List
from lime.lime_tabular import LimeTabularExplainer

# ============================================
# 1. LOAD MODEL AND SCALER
# ============================================

# Load training data and feature names (for reference in API)
X_train, y_train, feature_names = joblib.load('..\\models\\training_data_and_features.joblib')

# Load the trained model
# IMPORTANT: This must match EXACTLY the model architecture used during training
model = nn.Sequential(
    nn.Linear(len(feature_names), 128),  # Input: number of features
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Load saved weights
model.load_state_dict(torch.load('..\\models\\enriched_model.pt'))
model.eval()  # Set to evaluation mode

# Load scaler
scaler = joblib.load('..\\models\\scaler.joblib')

# Shap values 
shap_values = joblib.load('..\\models\\enriched_shap_values.joblib')

# Create a mapping of feature names to SHAP values for easier reference in the API
if isinstance(shap_values, list):
    # We focus on the positive class
    sv = shap_values[1]
else:
    # In some versions/configs, it might be a single 3D array
    sv = shap_values

# Compute Mean Absolute SHAP (Global Importance)
global_importance = np.abs(sv).mean(axis=0)
shap_feature_importance = dict(zip(feature_names, global_importance))

# Lime explainer creation using training data (for later use in the API)
lime_explainer = LimeTabularExplainer(
    training_data=X_train.to_numpy(),
    feature_names=feature_names,
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

def predict_fn(x):
    """Prediction function for LIME that returns probabilities"""
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        outputs = model(x_tensor)
        return np.hstack((1 - outputs.numpy(), outputs.numpy()))  # Return probabilities for both classes

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
            "predict_with_threshold": "/predict_with_threshold (POST)",
            "explain_global_predictions": "/explain_global_predictions (GET)",  
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
    Input: PatientData
    Output: Prediction (Diabetes/No Diabetes) with probability
    """
    try:
        # Validate input data
        data.validate(feature_names)

        # Extract features dynamically based on feature_names
        features = np.array([getattr(data, feature) for feature in feature_names]).reshape(1, -1)

        # Scale features
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
        
        # Generate LIME explanation
        lime_values = lime_explainer.explain_instance(
            data_row=features_scaled.flatten(),
            predict_fn=predict_fn
        ).as_list()
        
        lime_values = lime_values[:5]  # Limit to top 5 features for response

        return PredictionResponse(
            probability=round(probability, 4),
            prediction=prediction,
            threshold=threshold,
            confidence=round(confidence, 4),
            lime_values=lime_values
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(data_list: List[PatientData]):
    """
    Predict diabetes for multiple patients at once
    Input: List of PatientData (max 100)
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
            try:
                # Validate input data
                data.validate(feature_names)

                # Extract features dynamically based on feature_names
                features = np.array([getattr(data, feature) for feature in feature_names]).reshape(1, -1)

                # Scale features
                features_scaled = scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

                with torch.no_grad():
                    output = model(features_tensor)
                    probability = output.item()

                threshold = 0.5
                prediction = "Diabetes" if probability >= threshold else "No Diabetes"
                
                # Generate LIME explanation
                lime_values = lime_explainer.explain_instance(
                    data_row=features_scaled.flatten(),
                    predict_fn=predict_fn
                ).as_list()
                
                lime_values = lime_values[:5]  # Limit to top 5 features for response
                
                predictions.append({
                    "patient_id": i,
                    "probability": round(probability, 4),
                    "prediction": prediction,
                    "confidence": round(max(probability, 1 - probability), 4),
                    "lime_values": lime_values
                })

            except ValueError as ve:
                predictions.append({
                    "patient_id": i,
                    "error": str(ve)
                })
            except Exception as e:
                predictions.append({
                    "patient_id": i,
                    "error": str(e)
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
    - PatientData: Individual patient data
    - threshold: Decision threshold (0.0 to 1.0)
    """
    try:
        if threshold < 0 or threshold > 1:
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0 and 1"
            )

        # Validate input data
        data.validate(feature_names)

        # Extract features dynamically based on feature_names
        features = np.array([getattr(data, feature) for feature in feature_names]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Convert to tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            output = model(features_tensor)
            probability = output.item()

        prediction = "Diabetes" if probability >= threshold else "No Diabetes"

        # Generate LIME explanation
        lime_values = lime_explainer.explain_instance(
            data_row=features_scaled.flatten(),
            predict_fn=predict_fn
        ).as_list()
        
        lime_values = lime_values[:5]  # Limit to top 5 features for response
        
        return {
            "probability": round(probability, 4),
            "prediction": prediction,
            "threshold": threshold,
            "confidence": round(max(probability, 1 - probability), 4),
            "lime_values": lime_values
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain_global_predictions")
def explain_global_predictions():
    """
    Explain global predictions using SHAP values
    Output: SHAP values for each feature
    """
    # Convert NumPy arrays to Python scalars for JSON serialization
    top_shap_values = {
        key: float(value) if np.isscalar(value) else float(value.item())
        for key, value in sorted(shap_feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    }
    return {
        "shap_values": json.dumps(top_shap_values)
    }

# ============================================
# 5. RUN THE APP
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)