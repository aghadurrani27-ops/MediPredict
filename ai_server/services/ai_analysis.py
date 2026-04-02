import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set the key from .env for security, but keeping your logic flow
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_NEW_GROQ_API_KEY_HERE")

def generate_health_plan_service(user_id: str, goal: str, medical_context=None, lab_context=None):
    """
    Analyzes User ID, Goals, BMI, Disease Risks, and Lab Markers 
    to generate a medically-aware health plan.
    """
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, groq_api_key=GROQ_API_KEY)

    # --- Step 1: Data Preparation for the AI ---
    bmi = "Not Available"
    risks = "None detected"
    lab_summary = "No recent lab reports"

    if medical_context:
        res = medical_context.get("results", {})
        bmi = res.get("summary", {}).get("calculated_bmi", "N/A")
        risks = res.get("disease_risks", {})

    if lab_context:
        # Extracting markers to a readable string for the AI
        markers = lab_context.get("lab_values", [])
        lab_summary = ", ".join([f"{m['marker']}: {m['value']} ({m['status']})" for m in markers])

    # --- Step 2: Professional Prompt Engineering ---
    template = """
    You are a Senior Clinical Health Strategist for the MediPredict Platform.
    Generate a precision health plan based on the following specific user data:
    
    - User Goal: {goal}
    - BMI: {bmi}
    - Calculated Disease Risks: {risks}
    - Lab Report Markers: {lab_summary}

    STRICT RULES:
    1. If Heart Risk is 'High', reduce sodium and saturated fats.
    2. If BMI > 30, ensure a calorie deficit but keep protein high (1.6g/kg).
    3. If Lab Markers show 'High' Glucose, strictly limit refined sugars.
    4. Return ONLY a valid JSON object. No prose. No markdown.

    {{
      "goal": "{goal}",
      "daily_targets": {{ "calories": 2000, "protein_g": 150, "carbs_g": 250, "fat_g": 70 }},
      "meal_plan": [
          {{ "meal_type": "Breakfast", "time": "08:00 AM", "items": ["Item 1"], "calories": 400 }}
      ],
      "lifestyle_schedule": [
          {{ "task_name": "Exercise", "time": "07:00 AM", "category": "Workout", "duration_minutes": 30 }}
      ],
      "supplements": [
          {{ "name": "Vitamin", "dosage": "50mg", "timing": "Daily", "benefit": "Immunity" }}
      ]
    }}
    """

    prompt = PromptTemplate(
        input_variables=["goal", "bmi", "risks", "lab_summary"], 
        template=template
    )
    
    chain = prompt | llm

    try:
        # Step 3: Invoke AI with Context
        response = chain.invoke({
            "goal": goal,
            "bmi": bmi,
            "risks": json.dumps(risks),
            "lab_summary": lab_summary
        })

        raw_text = response.content.strip()
        
        # Robust JSON cleaning
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        
        return json.loads(raw_text.strip())
        
    except Exception as e:
        print(f" AI Generation Error: {e}")
        # Return a safe default to keep the app running
        return {{
            "goal": goal,
            "daily_targets": {{ "calories": 2000, "protein_g": 100, "carbs_g": 200, "fat_g": 60 }},
            "meal_plan": [], "lifestyle_schedule": [], "supplements": []
        }}