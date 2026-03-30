import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("1. Loading dataset...")

df = pd.read_csv('datasets/nutrition/nutrition_data.csv')

features = ['BMI', 'Chronic_Disease', 'Blood_Sugar_Level', 'Cholesterol_Level']
X = df[features].copy()
Y = df['Recommended_Meal_Plan'].copy()

print("2. Cleaning data...")
X.fillna('Normal', inplace=True)
Y.fillna('Balanced Diet', inplace=True)

encoder = {}
for col in features:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoder[col] = le
        
le_Y = LabelEncoder()
Y_encoded = le_Y.fit_transform(Y.astype(str))
encoder['target_encoder'] = le_Y

print("3. Training the model")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, Y_encoded)

print("4. Saving the model")
os.makedirs('ai_server/models', exist_ok=True)

joblib.dump({'model': model, 'encoders': encoder}, 'ai_server/models/nutrition_model.pkl')
