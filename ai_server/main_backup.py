import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
import jwt
import datetime
import shutil
import io
import numpy as np
from PIL import Image
from bson import ObjectId
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

# ML Imports
import tensorflow as tf
import joblib
from utils.image_preprocess import preprocess_image
from utils.ocr_engine import extract_medical_data

app = FastAPI(title="MediPredict API - Complete")

# DATABASE & FOLDERS
MONGO_DETAILS = "mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@cluster0.qbnmt8r.mongodb.net/?appName=Cluster0"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.medipredict
users_collection = database.get_collection("users")
screenings_collection = database.get_collection("screening")

UPLOAD_DIR = "upload_images"
REPORT_DIR = "uploaded_reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

#  SECURITY
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "medipredict_secret_key"
security = HTTPBearer()

#  LOAD ML MODELS (Loads once at startup) 
print("Loading ML Models...")
skin_model = tf.keras.models.load_model("models/skin_model.h5")
nails_model = tf.keras.models.load_model("models/nail_model.h5")
eyes_tongue_model = tf.keras.models.load_model("models/eyes_model.h5")

heart_model = joblib.load("models/heart.pkl")
diabetes_model = joblib.load("models/diabetes.pkl")
kidney_model = joblib.load("models/kidney.pkl")

try:
    nutrition_data = joblib.load("models/nutrition_model.pkl")
    nutrition_model = nutrition_data['model']
    nutrition_encoders = nutrition_data['encoders']
    print("✅ Nutrition model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load nutrition model: {e}")

print("All models loaded successfully!")


SKIN_CLASSES = {0: "Healthy Skin", 1: "Eczema", 2: "Melanoma"} 
NAIL_CLASSES = {0: "Healthy Nail", 1: "Nail Fungus", 2: "Psoriasis"}
EYES_TONGUE_CLASSES = {0: "Healthy Eye", 1: "Jaundice", 2: "Dehydrated Tongue"}

# PYDANTIC MODELS
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: int 
    gender: str
    bmi: float
    role: str = "patient"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ScreeningData(BaseModel):
    screening_type: str 
    symptoms: list[str]
    sleep_hours: float
    water_intake_liters: float

class ClinicalData(BaseModel):
    features: list[float]

class NutritionRequest(BaseModel):
    bmi: float
    chronic_disease: str
    blood_sugar: str
    cholesterol_level: str

#  HELPER FUNCTIONS 
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token.")

# AUTH APIS
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserRegister):
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_dict = user.dict()
    user_dict["password"] = get_password_hash(user.password)
    await users_collection.insert_one(user_dict)
    return {"message": "User registered successfully!"}

@app.post("/api/auth/login")
async def login_user(user: UserLogin):
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid Email or Password")
    token = create_access_token(data={"sub": db_user["email"]})
    return {"access_token": token, "token_type": "bearer"}

#  SCREENING & UPLOAD APIS 
@app.post("/api/screening/submit")
async def submit_screening(data: ScreeningData, current_email: str = Depends(get_current_user)):
    if data.screening_type not in ["skin", "nails", "eyes_tongue"]:
        raise HTTPException(status_code=400, detail="Invalid screening type.")
    
    screening_dict = data.dict()
    screening_dict["user_email"] = current_email 
    screening_dict["timestamp"] = datetime.datetime.utcnow()
    screening_dict["status"] = "Pending AI Analysis"
    
    result = await screenings_collection.insert_one(screening_dict)
    return {"message": "Saved!", "screening_id": str(result.inserted_id)}

@app.post("/api/screening/{screening_id}/upload-image")
async def upload_image(screening_id: str, file: UploadFile = File(...), current_email: str = Depends(get_current_user)):
    file_location = f"{UPLOAD_DIR}/{screening_id}_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    await screenings_collection.update_one({"_id": ObjectId(screening_id)}, {"$set": {"image_path": file_location}})
    return {"message": "Image uploaded!"}

@app.post("/api/screening/{screening_id}/upload-report")
async def upload_report(screening_id: str, file: UploadFile = File(...), current_email: str = Depends(get_current_user)):
    file_location = f"{REPORT_DIR}/{screening_id}_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    await screenings_collection.update_one({"_id": ObjectId(screening_id)}, {"$set": {"report_path": file_location}})
    return {"message": "Report uploaded!"}

@app.get("/api/screening/history")
async def get_user_history(current_email: str = Depends(get_current_user)):
    cursor = screenings_collection.find({"user_email": current_email})
    screenings = await cursor.to_list(length=100)
    for s in screenings:
        s["_id"] = str(s["_id"])
    return {"history": screenings}

def generate_lifestyle_plan(risk_level, cv_result, symptoms, ocr_data):
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    
    template = """
    You are an AI Health Coach for the MediPredict app. Based on the following user screening data,
    provide a short, personalized nutrition and lifestyle plan.

    CRITICAL RULES:
    1. Do NOT provide a medical diagnosis or prescribe medicine.
    2. Focus ONLY on preventive healthcare, diet, and lifestyle guidance.
    3. If the risk level is "high", your first sentence MUST strongly advise consulting a doctor.

    User Data:
    - Overall Risk Level: {risk_level}
    - Symptoms Reported: {symptoms}
    - AI Image Prediction: {cv_result}
    - Blood Report Alerts: {ocr_data}

    Format your response as 3 concise bullet points.
    """

    prompt = PromptTemplate(
        input_variables=["risk_level", "symptoms", "cv_result", "ocr_data"],
        template=template
    )

    chain = prompt | llm
    
    try:
        formatted_symptoms = ", ".join(symptoms) if symptoms else "None"
        formatted_ocr = str(ocr_data) if ocr_data else "None uploaded"
        
        response = chain.invoke({
            "risk_level": risk_level,
            "symptoms": formatted_symptoms,
            "cv_result": cv_result,
            "ocr_data": formatted_ocr
        })
        return response.content
    except Exception as e:
        print(f"Groq LLM Error: {e}")
        return "Please maintain a balanced diet, stay hydrated, and consult a doctor if you feel unwell."

def calculate_overall_risk(cv_result, ocr_data, symptoms):
    risk_level = "low" 
    risk_factors = 0
    
    if cv_result and "Healthy" not in cv_result:
        risk_factors += 2
        
    if ocr_data:
        for item in ocr_data:
            if item.get("status") != "Normal":
                risk_factors += 1
                
    if symptoms and len(symptoms) >= 3:
        risk_factors += 1
                
    if risk_factors >= 3:
        risk_level = "high"
    elif risk_factors >= 1:
        risk_level = "medium"
        
    return risk_level           

# THE REAL AI ANALYSIS API 
@app.post("/api/screening/{screening_id}/analyze")
async def analyze_screening(screening_id: str, current_email: str = Depends(get_current_user)):
    screening = await screenings_collection.find_one({"_id": ObjectId(screening_id)})
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    image_path = screening.get("image_path")
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="No image found to analyze.")

    # 1. Load and process the image from the disk
    img = Image.open(image_path).convert("RGB")
    processed_img = preprocess_image(img)

    # 2. Pick the right model based on screening type
    scr_type = screening.get("screening_type")
    
    try:
        if scr_type == "skin":
            prediction = skin_model(processed_img, training=False)
            class_idx = int(np.argmax(prediction))
            result_label = SKIN_CLASSES.get(class_idx, "Unknown Skin Condition")
        
        elif scr_type == "nails":
            prediction = nails_model(processed_img, training=False)
            class_idx = int(np.argmax(prediction))
            result_label = NAIL_CLASSES.get(class_idx, "Unknown Nail Condition")
            
        elif scr_type == "eyes_tongue":
            prediction = eyes_tongue_model(processed_img, training=False)
            class_idx = int(np.argmax(prediction))
            result_label = EYES_TONGUE_CLASSES.get(class_idx, "Unknown Eye/Tongue Condition")
            
        confidence_score = float(np.max(prediction)) * 100

        # 3.Run OCR if a blood report was uploaded
        ocr_results = None
        report_path = screening.get("report_path")
        if report_path and os.path.exists(report_path):
            ocr_results = extract_medical_data(report_path)
        
        overall_risk = calculate_overall_risk(
            cv_result=result_label,
            ocr_data=ocr_results,
            symptoms=screening.get("symptoms", [])
        )

        ai_recommendation = generate_lifestyle_plan(
            risk_level=overall_risk,
            cv_result=result_label,
            symptoms=screening.get("symptoms", []),
            ocr_data=ocr_results
        )
        
        # 4. Save actual results to Database
        ai_result = {
            "status": "Analysis Complete",
            "overall_risk_level": overall_risk,
            "predicted_condition": result_label,
            "ai_confidence": round(confidence_score, 2),
            "ocr_report_data": ocr_results,
            "ai_coach_plan": ai_recommendation
        }
        await screenings_collection.update_one({"_id": ObjectId(screening_id)}, {"$set": ai_result})

        return {"message": "AI Analysis complete!", "results": ai_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

#  TABULAR DATA APIS (Heart, Kidney, Diabetes)
@app.post("/predict/heart")
async def predict_heart(data: ClinicalData, current_email: str = Depends(get_current_user)):
    prediction = heart_model.predict([data.features])
    return {"prediction": int(prediction[0])}

@app.post("/predict/diabetes")
async def predict_diabetes(data: ClinicalData, current_email: str = Depends(get_current_user)):
    prediction = diabetes_model.predict([data.features])
    return {"prediction": int(prediction[0])}

@app.post("/predict/kidney")
async def predict_kidney(data: ClinicalData, current_email: str = Depends(get_current_user)):
    prediction = kidney_model.predict([data.features])
    return {"prediction": int(prediction[0])}

@app.get("/api/doctor/high-risk-cases")
async def get_high_risk_cases(current_email: str = Depends(get_current_user)):
    user = await users_collection.find_one({"email": current_email})
    if not user or user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Access denied. Doctors only.")
    
    cursor = screenings_collection.find({"overall_risk_level": "high"})
    high_risk_cases = await cursor.to_list(length=100)

    for case in high_risk_cases:
        case["_id"] = str(case["_id"])
        
    return {"status": "success", "count": len(high_risk_cases), "cases": high_risk_cases}

# Nurtition 
@app.post("/predict/nutrition", tags=["Predictions"])
async def predict_nutrition(req: NutritionRequest):
    try:
        import pandas as pd 
        input_data = pd.DataFrame([{
            'bmi': req.bmi,
            'chronic_disease': req.chronic_disease,
            'Blood_Sugar': req.blood_sugar,
            'Cholester_Level': req.cholesterol_level
        }])

        for col in input_data.columns:
            if col in nutrition_encoders:
                le = nutrition_encoders[col]
                input_data[col] = le.transform(input_data[col].astype(str))

        prediction_encoded = nutrition_model.predict(input_data)[0]
        recommended_diet = nutrition_encoders['target_encoder'].inverse_transform([prediction_encoded])[0]
        
        return {
            "status": "success",
            "recommended_meal_plan": recommended_diet
        }

    except Exception as e:
        return {"error": f"prediction failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)