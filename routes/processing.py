from fastapi import APIRouter, HTTPException, UploadFile, File
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import os
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploaded_datasets")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/train-gaussian-nb/")
async def train_gaussian_nb(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load the dataset
        if file_path.suffix != ".npz":
            raise HTTPException(
                status_code=400, detail="Only .npz files are supported."
            )

        data = np.load(file_path)

        if "data" not in data or "label" not in data:
            raise HTTPException(
                status_code=400,
                detail="The .npz file must contain 'data' and 'label' keys.",
            )

        # Reshape data and extract labels
        X = data["data"].reshape(data["data"].shape[0], -1)
        y = data["label"]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train GaussianNB
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        metrics = {
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        return {"message": "Model trained successfully!", "metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
