from fastapi import APIRouter

router = APIRouter()

# Define available models
available_models = {"1": "GaussianNB"}


@router.get("/")
async def get_available_models():
    """
    Returns the list of available models with their IDs and names.
    """
    try:
        return {
            "models": [
                {"id": model_id, "name": name}
                for model_id, name in available_models.items()
            ]
        }
    except Exception as e:
        return {"error": f"Failed to retrieve models: {str(e)}"}
