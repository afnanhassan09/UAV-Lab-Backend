from fastapi import APIRouter
from pathlib import Path
import hashlib

# Router instance
router = APIRouter()

# Directory containing uploaded files
UPLOAD_DIR = Path("data/uploaded_files")


def generate_file_id(file_path: Path) -> str:
    """
    Generate a consistent unique ID based on the file name and path.
    """
    hash_object = hashlib.sha256()
    hash_object.update(str(file_path.name).encode("utf-8"))
    return hash_object.hexdigest()


@router.get("/")
async def list_datasets():
    """
    Get all datasets available in the upload directory.
    Returns a JSON response with file information.
    """
    if not UPLOAD_DIR.exists():
        return {"message": "No datasets available."}

    datasets = []
    for file in UPLOAD_DIR.iterdir():
        if file.is_file():
            file_id = generate_file_id(file)
            file_type = file.suffix.lower()
            datasets.append({"id": file_id, "name": file.name, "type": file_type})

    return {"datasets": datasets}
