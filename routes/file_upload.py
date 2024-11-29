from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import os

# Router instance
router = APIRouter()

# Directory to save uploaded files
UPLOAD_DIR = Path("data/uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def handle_csv(file_path: Path):
    """
    Process CSV files using pandas.
    """
    try:
        data = pd.read_csv(file_path)
        return {"rows": data.shape[0], "columns": data.shape[1], "file_type": "CSV"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing CSV file: {str(e)}"
        )


def handle_hdf5(file_path: Path):
    """
    Process HDF5 files using h5py.
    """
    try:
        with h5py.File(file_path, "r") as h5_file:
            datasets = list(h5_file.keys())
        return {"datasets": datasets, "file_type": "HDF5"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing HDF5 file: {str(e)}"
        )


def handle_npz(file_path: Path):
    """
    Process NPZ files using numpy.
    """
    try:
        data = np.load(file_path)
        arrays = list(data.keys())
        return {"arrays": arrays, "file_type": "NPZ"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing NPZ file: {str(e)}"
        )


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, save it locally, and process it based on its format.
    """
    # Validate file extension
    allowed_extensions = [".csv", ".h5", ".npz"]
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV, HDF5 (.h5), or NPZ files are allowed.",
        )

    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process the file based on its format
    if file_extension == ".csv":
        return {
            "message": "File uploaded and processed successfully",
            "details": handle_csv(file_path),
        }
    elif file_extension == ".h5":
        return {
            "message": "File uploaded and processed successfully",
            "details": handle_hdf5(file_path),
        }
    elif file_extension == ".npz":
        return {
            "message": "File uploaded and processed successfully",
            "details": handle_npz(file_path),
        }
