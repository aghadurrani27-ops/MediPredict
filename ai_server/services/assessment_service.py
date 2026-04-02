import numpy as np
from services.model_loader import ml_assets
from services.vision_service import predict_image_condition

def predict_health_risks(data):
    """
    Processes user data (Basic Info + Symptoms + Images) 
    and returns a master professional health report.
    """
    try:
        basic = data.basic_info
        
        gender_map = {"Male": 1, "Female": 0, "Other": 0}
        gender_encoded = gender_map.get(basic.gender, 0)

        height_m = basic.height_cm / 100
        bmi = round(basic.weight_kg / (height_m ** 2), 1)

        features = [basic.age, bmi, gender_encoded]

        heart_risk = ml_assets.models['heart'].predict([features])[0]
        kidney_risk = ml_assets.models['kidney'].predict([features])[0]
        diabetes_risk = ml_assets.models['diabetes'].predict([features])[0]

        # 4. Computer Vision Analysis (Tongue, Eyes, Skin, Nails)
        vision_results = {
            "tongue": predict_image_condition(data.tongue_image_url, 'tongue'),
            "eyes": predict_image_condition(data.eyes_image_url, 'eyes'),
            "skin": predict_image_condition(data.skin_image_url, 'skin'),
            "nails": predict_image_condition(data.nails_image_url, 'nails'),
        }

        risks_count = int(heart_risk) + int(kidney_risk) + int(diabetes_risk)
        
        for condition in vision_results.values():
            if condition not in ["Healthy", "Not Provided", "Analysis Failed"]:
                risks_count += 1

        health_score = max(0, 100 - int(risks_count * 20))

        return {
            "summary": {
                "overall_health_score": health_score,
                "calculated_bmi": bmi,
                "risk_count": risks_count
            },
            "disease_risks": {
                "heart_disease": "High Risk" if heart_risk == 1 else "Low Risk",
                "kidney_disease": "High Risk" if kidney_risk == 1 else "Low Risk",
                "diabetes": "High Risk" if diabetes_risk == 1 else "Low Risk",
            },
            "visual_analysis": vision_results,
            "status": "Comprehensive Analysis Complete"
        }

    except Exception as e:
        print(f" Prediction Service Error: {e}")
        raise e