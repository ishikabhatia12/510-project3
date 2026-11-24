# AI helped generate this code.

# 1. Import packages
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, FiniteFloat


# 2. Read settings from the environment or a local .env file
load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
model_name = os.getenv("MLFLOW_MODEL_NAME", "student_outcome_model")


# 3. Create the API and load the champion model from MLflow
app = FastAPI()
mlflow.set_tracking_uri(tracking_uri)
model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
champion = mlflow.MlflowClient().get_model_version_by_alias(model_name, "champion")
confidence_threshold = float(champion.tags.get("confidence_threshold", 0.4))


# 4. List the two numbers the API expects in the JSON request
class StudentGrades(BaseModel):
    admission_grade: FiniteFloat
    previous_qualification_grade: FiniteFloat


# 5. Make a prediction when someone sends grades to /predict
@app.post("/predict")
def predict(grades: StudentGrades):
    # Create one row with the same column names and order as in model.py
    student = pd.DataFrame({
        "Admission grade": [grades.admission_grade],
        "Previous qualification (grade)": [grades.previous_qualification_grade],
    })

    probabilities = model.predict_proba(student)[0]
    best = probabilities.argmax()
    confidence = float(probabilities[best])

    if confidence < confidence_threshold:
        return {"prediction": "uncertain", "confidence": confidence}

    return {"prediction": str(model.classes_[best]), "confidence": confidence}