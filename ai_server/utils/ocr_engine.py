import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_medical_data(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    
    # Looks for "Hemoglobin" or "Hb" followed by a number
    hb_pattern = r"(?:Hemoglobin|Hb)\s*[:\-]?\s*(\d+\.?\d*)"
    sugar_pattern = r"(?:Glucose|Sugar)\s*[:\-]?\s*(\d+\.?\d*)"
    
    hb_match = re.search(hb_pattern, text, re.IGNORECASE)
    sugar_match = re.search(sugar_pattern, text, re.IGNORECASE)
    
    return {
        "hemoglobin": float(hb_match.group(1)) if hb_match else None,
        "glucose": float(sugar_match.group(1)) if sugar_match else None,
        "raw_text": text
    }