
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="Kyphosis Prediction API", version="1.0")

model = joblib.load("Kyphosis.pkl")
mlflow.set_experiment("Kyphosis_experiment")
# ----------------------------
# Input schema
# ----------------------------
class KyphosisInput(BaseModel):
    Age: int
    Number: int
    Start: int

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Kyphosis Prediction API is live"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(data: KyphosisInput):
    # Prepare input for model
    input_data = np.array([[data.Age, data.Number, data.Start]])

    # Make prediction
    prediction = model.predict(input_data)[0]  # scalar 0 or 1


    # Log input & prediction to MLflow
    try:
        with mlflow.start_run(run_name="Inference_Run"):
            # Log inputs
            mlflow.log_params({
                "Age": data.Age,
                "Number": data.Number,
                "Start": data.Start
            })
    except Exception as e:
        print("MLflow logging error:", e)

    # Return prediction as boolean
    return {"Kyphosis": prediction}




