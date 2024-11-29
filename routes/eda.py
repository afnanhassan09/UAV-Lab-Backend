import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json

router = APIRouter()

UPLOAD_DIR = Path("data/uploaded_files")
STATIC_IMAGE_DIR = Path("static/images")
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)
OUTPUT_MAIN_FOLDER = "ALFA Dataset/concatenated_trajectories"


def load_dataset(file_path: Path) -> pd.DataFrame:
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
            data = dict(np.load(file_path))
            return pd.DataFrame(next(iter(data.values())))
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")


def generate_file_id(file_path: Path) -> str:
    """
    Generate a consistent unique ID based on the file name.
    """
    import hashlib

    hash_object = hashlib.sha256()
    hash_object.update(str(file_path.name).encode("utf-8"))
    return hash_object.hexdigest()


def replace_nan_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN values in the DataFrame with None for JSON serialization.
    """
    return df.applymap(lambda x: None if pd.isna(x) else x)


def save_plot_as_image(plot_func, file_name: str):
    """
    Utility to save a plot as an image.
    """
    file_path = STATIC_IMAGE_DIR / file_name
    plot_func()
    plt.savefig(file_path)
    plt.close()
    return str(file_path)


@router.get("/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    try:
        if not UPLOAD_DIR.exists():
            raise HTTPException(status_code=404, detail="No datasets available.")

        # Locate the file based on the dataset_id
        file_path = None
        for file in UPLOAD_DIR.iterdir():
            if file.is_file() and generate_file_id(file) == dataset_id:
                file_path = file
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        # Load the dataset
        try:
            df = load_dataset(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset: {str(e)}"
            )

        # Filter out non-numeric columns for analysis
        df_numeric = df.select_dtypes(include=[np.number])

        # Replace NaN values with None for JSON serialization
        df = replace_nan_with_none(df)

        # --- 1. Histograms ---
        histograms = {}
        for col in df_numeric.columns:
            hist_file_name = f"{dataset_id}_{col}_histogram.png"
            histograms[col] = save_plot_as_image(
                lambda: sns.histplot(df_numeric[col], kde=True), hist_file_name
            )

        # --- 2. Correlation Analysis ---
        correlation_matrix = df_numeric.corr().fillna(0).to_dict()

        # --- 3. Trend, Seasonality, and Anomaly Detection ---
        trend_analysis = {}
        trend_image_url = None
        if "timestamp" in df.columns and "value" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.set_index("timestamp", inplace=True)

            decomposition = seasonal_decompose(
                df["value"].dropna(), model="additive", period=30
            )
            trend_analysis["trend"] = decomposition.trend.dropna().tolist()
            trend_analysis["seasonality"] = decomposition.seasonal.dropna().tolist()

            df["zscore"] = zscore(df["value"].dropna())
            anomalies = (
                df[df["zscore"].abs() > 3][["value"]]
                .reset_index()
                .to_dict(orient="records")
            )
            trend_analysis["anomalies"] = anomalies

            trend_image_url = save_plot_as_image(
                lambda: plt.plot(decomposition.trend, label="Trend", color="blue"),
                f"{dataset_id}_trend_analysis.png",
            )

        # --- 4. Missing Value Analysis ---
        missing_summary = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()

        # --- 5. Descriptive Statistics ---
        descriptive_stats = df.describe(include="all").fillna("N/A").to_dict()

        # --- EDA Reports ---
        automated_report = {
            "correlation_analysis": correlation_matrix,
            "anomalies": trend_analysis.get("anomalies", []),
            "missing_summary": missing_summary,
            "missing_percentage": missing_percentage,
            "descriptive_statistics": descriptive_stats,
        }

        # --- Save Missing Values and Descriptive Stats to Reports ---
        output_folder = os.path.join(OUTPUT_MAIN_FOLDER, dataset_id)
        os.makedirs(output_folder, exist_ok=True)

        # Save missing value report
        missing_report_path = os.path.join(
            output_folder, f"{dataset_id}_missing_values_report.json"
        )
        with open(missing_report_path, "w") as f:
            json.dump(
                {
                    "missing_summary": missing_summary,
                    "missing_percentage": missing_percentage,
                },
                f,
            )

        # Save descriptive statistics
        descriptive_stats_path = os.path.join(
            output_folder, f"{dataset_id}_descriptive_stats.json"
        )
        with open(descriptive_stats_path, "w") as f:
            json.dump(descriptive_stats, f)

        # --- Response ---
        return {
            "dataset_id": dataset_id,
            "histograms": {
                col: f"/static/images/{Path(url).name}"
                for col, url in histograms.items()
            },
            "correlation_analysis": correlation_matrix,
            "trend_analysis": {
                **trend_analysis,
                "trend_image": (
                    f"/static/images/{Path(trend_image_url).name}"
                    if trend_image_url
                    else None
                ),
            },
            "automated_report": automated_report,
            "missing_values_report": missing_report_path,
            "descriptive_stats_report": descriptive_stats_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
