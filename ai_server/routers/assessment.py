from config.db import assessments_collection
from fastapi import APIRouter, HTTPException
from models.assessment import AssessmentSubmitRequest, AssessmentResponse
from services.assessment_service import predict_health_risks 

router = APIRouter()

@router.post("/submit", response_model=AssessmentResponse)
async def submit_assessment(data: AssessmentSubmitRequest):
    try:
        ml_results = predict_health_risks(data.basic_info, data.symptoms)

        return AssessmentResponse(
            status="success",
            message="Comprehensive health assessment completed.",
            user_id=data.user_id,
            results=ml_results # Sending back Heart, Diabetes, Kidney, and Health Score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Prediction Error: {str(e)}")