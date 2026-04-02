from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import plan, assessment, ocr
from services.model_loader import ml_assets
from routers import auth



app = FastAPI(
    title="MediPredict Professional API",
    description="Backend for the AI-powered health screening and planning app.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the MediPredict API. System is running optimally."}

app.include_router(plan.router, prefix="/api/v1/plan", tags=["Health Plan Generation"])
app.include_router(assessment.router, prefix="/api/v1/assessment", tags=["User Assessment"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["User Authentication"])
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["Lab OCR Processing"])
