import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- 1. HEALTH PLAN GENERATION LOGIC ---
def generate_health_plan_service(user_id: str, goal: str):
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

    template = """
    You are an expert AI Health Coach for MediPredict.
    Generate a comprehensive daily health plan for a user with the primary goal: {goal}.
    Return the response STRICTLY as a valid JSON object.
    
    Structure:
    {{
      "goal": "{goal}",
      "daily_targets": {{ "calories": 2000, "protein_g": 150, "carbs_g": 250, "fat_g": 70 }},
      "meal_plan": [{{ "meal_type": "Breakfast", "time": "08:00 AM", "items": ["Oatmeal"], "calories": 400 }}],
      "lifestyle_schedule": [{{ "task_name": "Morning Walk", "time": "07:00 AM", "category": "Exercise", "duration_minutes": 30 }}],
      "supplements": [{{ "name": "Vitamin D", "dosage": "1000 IU", "timing": "Morning", "benefit": "Bone Health" }}]
    }}
    """

    prompt = PromptTemplate(input_variables=["goal"], template=template)
    chain = prompt | llm

    try:
        response = chain.invoke({"goal": goal})
        raw_text = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw_text)
    except Exception as e:
        print(f"Plan Generation Error: {e}")
        return {"error": "Failed to generate plan"}


# --- 2. NEW: OCR TEXT PARSING LOGIC ---
def parse_lab_text_with_ai(raw_text: str):
    """
    Takes messy text from Tesseract and uses Llama 3 to extract 
    structured medical values for the 'History' dashboard.
    """
    llm = ChatGroq(model="llama3-8b-8192", temperature=0) # Temp 0 for data accuracy

    template = """
    Analyze the following raw OCR text from a medical lab report. 
    Extract the biomarkers (like Glucose, Cholesterol, Hemoglobin, etc.).
    
    Return a JSON LIST of objects with these EXACT keys: 
    "marker", "value", "unit", "status", "reference_range".
    
    Raw Text: {text}
    
    If no markers are found, return an empty list []. 
    STRICTLY return JSON only. No prose, no conversation.
    """

    prompt = PromptTemplate(input_variables=["text"], template=template)
    chain = prompt | llm

    try:
        response = chain.invoke({"text": raw_text})
        # Clean response of markdown backticks
        clean_json = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        print(f"AI Parsing Error: {e}")
        return []