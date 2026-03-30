
def analyze_blood_values(extracted_data):
    results = {}
    
    if "hemoglobin" in extracted_data:
        hb = extracted_data["hemoglobin"]
        if hb < 13.5:
            results["hemoglobin"] = {"status": "Low", "advice": "Eat more iron-rich foods like spinach."}
        elif hb > 17.5:
            results["hemoglobin"] = {"status": "High", "advice": "Consult a doctor about polycythemia."}
        else:
            results["hemoglobin"] = {"status": "Normal", "advice": "Your levels look great!"}
            
    return results