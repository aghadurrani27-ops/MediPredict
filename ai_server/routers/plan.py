from fastapi import APIRouter, HTTPException
from models.plan import GeneratePlanRequest, ComprehensiveHealthPlan
from services.ai_analysis import generate_health_plan_service
from config.db import assessments_collection, lab_reports_collection # Critical for personalization

router = APIRouter()

@router.post("/generate", response_model=ComprehensiveHealthPlan)
async def generate_plan(request: GeneratePlanRequest):
    try:
        # 1. Fetch the LATEST Assessment (BMI, Risks, Vision) from DB
        latest_assessment = await assessments_collection.find_one(
            {"user_id": request.user_id},
            sort=[("timestamp", -1)] # Get the most recent one
        )

        # 2. Fetch the LATEST Lab Report (Blood Work) from DB
        latest_lab = await lab_reports_collection.find_one(
            {"user_id": request.user_id},
            sort=[("timestamp", -1)]
        )

        # 3. Pass ALL medical context to the AI Service
        # If no data is found, the service will handle it gracefully
        ai_plan_data = generate_health_plan_service(
            user_id=request.user_id, 
            goal=request.selected_goals[0] if request.selected_goals else "General Health",
            medical_context=latest_assessment,
            lab_context=latest_lab
        )
        
        return ai_plan_data
        
    except Exception as e:
        print(f"Plan Generation Route Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))