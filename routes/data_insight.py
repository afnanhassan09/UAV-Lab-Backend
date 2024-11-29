from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import hashlib
import numpy as np

# Router instance
router = APIRouter()

# Directory containing uploaded files
UPLOAD_DIR = Path("data/uploaded_files")


def generate_file_id(file_path: Path) -> str:
    """
    Generate a consistent unique ID based on the file name.
    """
    hash_object = hashlib.sha256()
    hash_object.update(str(file_path.name).encode("utf-8"))
    return hash_object.hexdigest()


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the dataset based on its file type.
    Supports: CSV, HDF5 (.h5), NPZ.
    """
    try:
        if file_path.suffix == ".csv":
            # Read CSV file into a DataFrame
            return pd.read_csv(file_path)
        elif file_path.suffix == ".h5":
            # Read HDF5 file into a DataFrame (default dataset)
            return pd.read_hdf(file_path)
        elif file_path.suffix == ".npz":
            # Load NPZ as a dictionary and convert the first array-like object to DataFrame
            data = dict(np.load(file_path))
            return pd.DataFrame(next(iter(data.values())))
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")


def replace_nan_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN values in the DataFrame with None for JSON serialization.
    """
    return df.applymap(lambda x: None if pd.isna(x) else x)


@router.get("/{dataset_id}")
async def get_data_insights(dataset_id: str):
    """
    Get data insights for the dataset identified by the unique ID.
    Returns:
    - Number of rows, columns, and data types.
    - Missing values and percentage.
    - Preliminary outlier and anomaly counts.
    - Mean for numerical features and most frequent values for categorical features.
    """
    # Locate the file based on the ID
    if not UPLOAD_DIR.exists():
        raise HTTPException(status_code=404, detail="No datasets available.")

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
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    # Replace NaN values with None
    df = replace_nan_with_none(df)

    # Generate Data Insights
    num_rows, num_columns = df.shape
    data_types = df.dtypes.to_dict()

    # Handle missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / num_rows * 100).round(2)

    # Outlier count for numerical columns
    numerical_df = df.select_dtypes(include=["number"])
    categorical_df = df.select_dtypes(exclude=["number"])
    outlier_count = (
        ((numerical_df - numerical_df.mean()).abs() > 3 * numerical_df.std())
        .sum()
        .sum()
    )

    # Summary statistics
    categorical_summary = {}

    for col in categorical_df.columns:
        mode = df[col].mode()
        most_frequent = mode.iloc[0] if not mode.empty else None
        count = (
            df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        )
        categorical_summary[col] = {
            "most_frequent": most_frequent,
            "count": int(count),  # Ensure conversion to a native Python int
        }

    numerical_summary = {}
    for col in numerical_df.columns:
        mean = df[col].mean()
        numerical_summary[col] = {
            "mean": (
                float(mean) if not np.isnan(mean) else None
            )  # Ensure conversion to Python float
        }
    feature_summary = {
        "numerical": numerical_summary,
        "categorical": categorical_summary,
    }

    # Construct the response dictionary
    response_data = {
        "file_name": file_path.name,
        "rows": num_rows,
        "columns": num_columns,
        "data_types": {col: str(dtype) for col, dtype in data_types.items()},
        "missing_values": missing_values.to_dict(),
        "missing_percentage": missing_percentage.to_dict(),
        "preliminary_outlier_count": int(outlier_count),
        "feature_summary": feature_summary,
    }

    # Return response data
    return response_data
