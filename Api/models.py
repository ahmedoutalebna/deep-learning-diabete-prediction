from pydantic import BaseModel, Field
from typing import List, Optional


# ============================================
# 3. DEFINE REQUEST/RESPONSE MODELS
# ============================================

# "GenHlth", "HighBP", "BMI", "DiffWalk", "HighChol", "Age", "HeartDiseaseorAttack", "PhysHlth", "Income", "Education"

class PatientData(BaseModel):
    """
    Input model for patient health indicators with explicit fields.
    All fields are optional, but validation ensures required fields are present.
    """
    HighBP: Optional[float] = Field(None, description="High Blood Pressure (1 for Yes, 0 for No)")
    HighChol: Optional[float] = Field(None, description="High Cholesterol (1 for Yes, 0 for No)")
    CholCheck: Optional[float] = Field(None, description="Cholesterol Check (1 for Yes, 0 for No)")
    BMI: Optional[float] = Field(None, description="Body Mass Index")
    Smoker: Optional[float] = Field(None, description="Smoking Status (1 for Yes, 0 for No)")
    Stroke: Optional[float] = Field(None, description="History of Stroke (1 for Yes, 0 for No)")
    PhysActivity: Optional[float] = Field(None, description="Physical Activity (1 for Yes, 0 for No)")
    Fruits: Optional[float] = Field(None, description="Fruit Consumption (1 for Yes, 0 for No)")
    Veggies: Optional[float] = Field(None, description="Vegetable Consumption (1 for Yes, 0 for No)")
    HvyAlcoholConsump: Optional[float] = Field(None, description="Heavy Alcohol Consumption (1 for Yes, 0 for No)")
    AnyHealthcare: Optional[float] = Field(None, description="Any Healthcare Coverage (1 for Yes, 0 for No)")
    NoDocbcCost: Optional[float] = Field(None, description="No Doctor Due to Cost (1 for Yes, 0 for No)")
    GenHlth: Optional[float] = Field(None, description="General Health(0-5 scale)")
    MentHlth: Optional[float] = Field(None, description="Mental Health (0-30 days)")
    PhysHlth: Optional[float] = Field(None, description="Physical Health (0-30 days)")
    DiffWalk: Optional[float] = Field(None, description="Difficulty Walking (1 for Yes, 0 for No)")
    Sex: Optional[float] = Field(None, description="Gender (1 for Male, 0 for Female)")
    Age: Optional[float] = Field(None, description="Age Group (1-13 scale)")
    Income: Optional[float] = Field(None, description="Income Level (1-8 scale)")
    Education: Optional[float] = Field(None, description="Education Level (1-6 scale)")
    HeartDiseaseorAttack: Optional[float] = Field(None, description="History of Heart Disease or Attack (1 for Yes, 0 for No)")

    def validate(self, required_features: list):
        """
        Validate that all required features are present.
        """
        missing_fields = [field for field in required_features if getattr(self, field, None) is None]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    class Config:
        json_schema_extra = {
            "example": {
                "HighBP": 1.0,  # High Blood Pressure (1 for Yes, 0 for No)
                "HighChol": 1.0,  # High Cholesterol (1 for Yes, 0 for No)
                "CholCheck": 1.0,  # Cholesterol Check (1 for Yes, 0 for No)
                "BMI": 25.0,  # Body Mass Index
                "Smoker": 0.0,  # Smoking Status (1 for Yes, 0 for No)
                "Stroke": 0.0,  # History of Stroke (1 for Yes, 0 for No)
                "Diabetes": 0.0,  # Diabetes Status (1 for Yes, 0 for No)
                "PhysActivity": 1.0,  # Physical Activity (1 for Yes, 0 for No)
                "Fruits": 1.0,  # Fruit Consumption (1 for Yes, 0 for No)
                "Veggies": 1.0,  # Vegetable Consumption (1 for Yes, 0 for No)
                "HvyAlcoholConsump": 0.0,  # Heavy Alcohol Consumption (1 for Yes, 0 for No)
                "AnyHealthcare": 1.0,  # Any Healthcare Coverage (1 for Yes, 0 for No)
                "NoDocbcCost": 0.0,  # No Doctor Due to Cost (1 for Yes, 0 for No)
                "GenHlth": 3.0,  # General Health (0-5 scale)
                "MentHlth": 5.0,  # Mental Health (0-30 days)
                "PhysHlth": 5.0,  # Physical Health (0-30 days)
                "DiffWalk": 0.0,  # Difficulty Walking (1 for Yes, 0 for No)
                "Sex": 1.0,  # Gender (1 for Male, 0 for Female)
                "Age": 8.0,  # Age Group (1-13 scale)
                "Income": 4.0,  # Income Level (1-8 scale)
                "Education": 4.0,  # Education Level (1-6 scale)
                "HeartDiseaseorAttack": 0.0  # History of Heart Disease or Attack (1 for Yes, 0 for No)
            }
        }

class PredictionResponse(BaseModel):
    """Output model for predictions"""
    probability: float
    prediction: str
    threshold: float
    confidence: float
    lime_values: List[tuple]

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[dict]
    count: int