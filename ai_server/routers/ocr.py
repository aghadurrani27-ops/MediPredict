import io
import pytesseract
import datetime
from PIL import Image
from pdf2image import convert_from_bytes
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.ocr import LabReportResponse
from services.plan_service import parse_lab_text_with_ai
from config.db import database # Added for Database persistence

router = APIRouter()
lab_collection = database.get_collection("lab_reports")

@router.post("/upload", response_model=LabReportResponse)
async def process_lab_report(user_id: str, file: UploadFile = File(...)):
    try:
        # 1. Validate the file type
        allowed_types = ["application/pdf", "image/jpeg", "image/png"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, JPEG, and PNG are allowed."
            )
        
        contents = await file.read()
        extracted_text = ""

        # 2. Extraction Logic
        if file.content_type in ["image/jpeg", "image/png"]:
            image = Image.open(io.BytesIO(contents))
            extracted_text = pytesseract.image_to_string(image)
            
        elif file.content_type == "application/pdf":
            pages = convert_from_bytes(contents)
            for page in pages:
                extracted_text += pytesseract.image_to_string(page) + "\n"

        # 3. AI Structuring Logic (Moved INSIDE the function)
        structured_lab_values = []
        if extracted_text.strip() and len(extracted_text) > 10:
            # This calls your Groq/Llama service to turn text into JSON
            structured_lab_values = parse_lab_text_with_ai(extracted_text)
        else:
            extracted_text = "Could not extract any text from document."

        # 4. Save to MongoDB (Professional History Tracking)
        report_doc = {
            "user_id": user_id,
            "filename": file.filename,
            "timestamp": datetime.datetime.utcnow(),
            "lab_values": structured_lab_values,
            "raw_text": extracted_text[:500] # Save a snippet of raw text
        }
        await lab_collection.insert_one(report_doc)

        # 5. Return Response
        return LabReportResponse(
            filename=file.filename,
            extracted_text=extracted_text.strip(),
            lab_values=structured_lab_values,
            status="success"
        )
    
    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR Processing Error: {str(e)}")
    
    finally:
        await file.close()