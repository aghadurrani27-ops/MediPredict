import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from services.model_loader import ml_assets

SKIN_CLASSES = {0: "Healthy", 1: "Eczema", 2: "Melanoma"}
EYE_CLASSES = {0: "Healthy", 1: "Jaundice", 2: "Cataract"}

def predict_image_condition(image_url: str, model_type: str):
    """
    Downloads an image from a URL, preprocesses it, and runs the specified CV model.
    """
    if not image_url:
        return "Not Provided"

    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        model = ml_assets.models.get(model_type)
        if not model:
            return "Model Not Loaded"

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        
        if model_type == 'skin':
            return SKIN_CLASSES.get(class_idx, "Unknown")
        elif model_type == 'eyes':
            return EYE_CLASSES.get(class_idx, "Unknown")
        
        return "Analysis Complete"

    except Exception as e:
        print(f"Vision Service Error ({model_type}): {e}")
        return "Analysis Failed"