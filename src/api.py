from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Student Success Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate and load the trained model
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "student_success_model.joblib"
model = joblib.load(model_path)


# Request body schema
class StudentInput(BaseModel):
    hours_studied: float
    attendance_rate: float
    past_grade_avg: float
    assignments_completed: int
    sleep_hours: float


@app.get("/")
def root():
    return {"message": "Student Success Predictor API is running"}


@app.post("/predict")
def predict_student_success(data: StudentInput):
    # Order of features must match training:
    # ["hours_studied", "attendance_rate", "past_grade_avg", "assignments_completed", "sleep_hours"]
    features = np.array([[
        data.hours_studied,
        data.attendance_rate,
        data.past_grade_avg,
        data.assignments_completed,
        data.sleep_hours,
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]  # probability of passing

    return {
        "pass_probability": round(float(prob), 3),
        "prediction": int(pred)  # 1 = pass, 0 = fail
    }
