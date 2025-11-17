from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    hours_studied = rng.normal(loc=3.5, scale=1.5, size=n_samples).clip(0, 10)
    attendance_rate = rng.uniform(0.4, 1.0, size=n_samples)  # 40%–100%
    past_grade_avg = rng.normal(loc=6.5, scale=1.2, size=n_samples).clip(4, 10)
    assignments_completed = rng.integers(5, 15, size=n_samples)
    sleep_hours = rng.normal(loc=7, scale=1.5, size=n_samples).clip(3, 10)

    # Simple rule + noise to generate labels (1 = pass, 0 = fail)
    score = (
        0.4 * hours_studied
        + 3.0 * attendance_rate
        + 0.5 * past_grade_avg
        + 0.2 * assignments_completed
        - 0.3 * (8 - sleep_hours)
    )
    # add noise
    score += rng.normal(0, 2.0, size=n_samples)

    pass_prob = 1 / (1 + np.exp(-0.3 * (score - score.mean())))
    y = (pass_prob > 0.5).astype(int)

    data = pd.DataFrame(
        {
            "hours_studied": hours_studied,
            "attendance_rate": attendance_rate,
            "past_grade_avg": past_grade_avg,
            "assignments_completed": assignments_completed,
            "sleep_hours": sleep_hours,
            "pass_exam": y,
        }
    )

    return data


def train_and_save_model():
    # 1) generate data
    df = generate_synthetic_data(n_samples=1500)

    X = df.drop(columns=["pass_exam"])
    y = df["pass_exam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) build pipeline: scaler + logistic regression
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 3) train
    pipeline.fit(X_train, y_train)

    # 4) evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    # 5) save model pipeline
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "student_success_model.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
