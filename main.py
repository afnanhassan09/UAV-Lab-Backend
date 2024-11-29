from fastapi import FastAPI
from routes.file_upload import router as file_upload_router
from routes.file_list import router as file_list_router
from routes.data_insight import router as data_insights_router
from routes.eda import router as eda_summary_router
from routes.processing import router as preprocessing_router
from routes.get_models import router as get_models_router
from routes.apply_model import router as apply_model_router

app = FastAPI()

# Include routes
app.include_router(file_upload_router, prefix="/file-upload")
app.include_router(file_list_router, prefix="/file-list")
app.include_router(data_insights_router, prefix="/get_insights")
app.include_router(eda_summary_router, prefix="/get_eda")
app.include_router(eda_summary_router, prefix="/get_models")
app.include_router(preprocessing_router, prefix="/preprocess")
app.include_router(get_models_router, prefix="/get_models")
app.include_router(apply_model_router, prefix="/apply_model")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Aviation Anomaly Detection Framework API"}
