import joblib
import tensorflow as tf
import os

class ModelLoader:
    def __init__(self):
        self.heart_model = None
        self.diabetes_model = None
        self.kidney_model = None
        self.vision_model = None
        self.load_models()

    def load_models(self):
        try:
            print("--- 🩺 Loading Medical AI Assets ---")
            
            # Use os.path.join for Windows/Linux compatibility
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_path, "models")

            # Load Tabular Models (Joblib)
            self.heart_model = joblib.load(os.path.join(models_dir, "heart_model.pkl"))
            self.diabetes_model = joblib.load(os.path.join(models_dir, "diabetes_model.pkl"))
            self.kidney_model = joblib.load(os.path.join(models_dir, "kidney_model.pkl"))
            
            # Load Vision Model (TensorFlow)
            # Use compile=False to avoid loading optimizer state if only predicting
            self.vision_model = tf.keras.models.load_model(
                os.path.join(models_dir, "medical_vision_v1.h5"), 
                compile=False
            )
            
            print("✅ Status: All models mapped to memory successfully.")
        except Exception as e:
            print(f" Critical Error loading models: {e}")

# --- MOVE THIS OUTSIDE THE CLASS ---
# This creates a single instance that other files can import
ml_assets = ModelLoader()