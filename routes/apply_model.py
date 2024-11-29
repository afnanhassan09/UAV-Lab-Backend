import hashlib
from fastapi import APIRouter, HTTPException
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os

router = APIRouter()

UPLOAD_DIR = Path("uploaded_datasets")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def generate_file_id(file_path: Path) -> str:
    """
    Generate a consistent unique ID based on the file name.
    """
    hash_object = hashlib.sha256()
    hash_object.update(str(file_path.name).encode("utf-8"))
    return hash_object.hexdigest()


def load_dataset(file_path: Path):
    """
    Load the dataset based on its file type.
    Supports: CSV, HDF5 (.h5), NPZ.
    """
    try:
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".h5":
            return pd.read_hdf(file_path)
        elif file_path.suffix == ".npz":
            data = np.load(file_path)
            if "data" not in data or "label" not in data:
                raise ValueError("The NPZ file must contain 'data' and 'label' keys.")
            return data["data"], data["label"]
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")


def detect_anomalies_kmeans(X, n_clusters=1, threshold_percentile=99):
    """
    Detect anomalies using K-means clustering.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    n_clusters : int, optional
        Number of clusters for K-means.
    threshold_percentile : int, optional
        Percentile to use for determining anomaly threshold.

    Returns:
    --------
    np.ndarray
        Anomalies as a boolean mask.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Calculate distances to cluster centers
    distances = kmeans.transform(X_scaled)
    labels = kmeans.labels_

    # Distance of each point from its cluster centroid
    distance_to_centroid = distances[np.arange(len(X_scaled)), labels]

    # Define a threshold for anomalies
    threshold = np.percentile(distance_to_centroid, threshold_percentile)
    anomalies = distance_to_centroid > threshold

    return anomalies


@router.get("/supervised/{model_id}/{dataset_id}")
async def train_model(model_id: str, dataset_id: str, train_test_ratio: float = 0.8):
    """
    Train a specified model using the dataset identified by dataset_id.
    Supported models: GaussianNB.
    Parameters:
        model_id (str): The ID of the model to use.
        dataset_id (str): The ID of the dataset to load.
        train_test_ratio (float): The ratio of the dataset to use for training (default: 0.8).
    """
    try:
        # Validate the train-test split ratio
        if not (0.0 < train_test_ratio < 1.0):
            raise HTTPException(
                status_code=400,
                detail="Train-test split ratio must be between 0 and 1 (exclusive).",
            )

        # Locate the dataset based on its ID
        file_path = None
        for file in UPLOAD_DIR.iterdir():
            if file.is_file() and generate_file_id(file) == dataset_id:
                file_path = file
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        # Load the dataset
        try:
            X, y = load_dataset(file_path)
            if isinstance(X, pd.DataFrame):  # Handle CSV/HDF5 where X is a DataFrame
                y = X.pop("label").values  # Assumes 'label' column exists
                X = X.values
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset: {str(e)}"
            )

        # Reshape data if it is 3D (specific to NPZ files)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        # Split the dataset
        test_size = 1 - train_test_ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train the selected model
        if model_id == "1":  # GaussianNB
            model = GaussianNB()
        else:
            raise HTTPException(
                status_code=400, detail=f"Model ID '{model_id}' not supported."
            )

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

        return {
            "message": f"Model '{model_id}' trained successfully!",
            "metrics": metrics,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unsupervised/{dataset_id}")
async def unsupervised_model_with_anomalies(
    dataset_id: str, train_test_ratio: float = 0.8
):
    """
    Detect anomalies using K-means clustering and then train a GaussianNB model using anomalies.

    Parameters:
    -----------
    dataset_id : str
        ID of the dataset to load.
    train_test_ratio : float, optional
        Ratio for train-test split (default is 0.8).

    Returns:
    --------
    dict
        Metrics and anomalies.
    """
    try:
        # Locate the dataset
        file_path = None
        for file in UPLOAD_DIR.iterdir():
            if file.is_file() and generate_file_id(file) == dataset_id:
                file_path = file
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        # Load the dataset
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

        # Ensure the dataset has numeric data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise HTTPException(
                status_code=400, detail="Dataset does not contain any numeric columns."
            )

        X = df[numeric_columns].values

        # Detect anomalies
        anomalies = detect_anomalies_kmeans(X)
        df["anomaly"] = anomalies.astype(int)

        # Use anomalies as a feature for GaussianNB
        y = df["anomaly"].values
        X = df[numeric_columns].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_test_ratio, random_state=42
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

        return {
            "message": "Unsupervised model with anomalies trained successfully!",
            "metrics": metrics,
            "anomaly_counts": {
                "total": len(anomalies),
                "detected": int(anomalies.sum()),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
