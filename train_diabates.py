import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("datasets/diabetes/diabetes.csv")

# Auto-separate
x = data.iloc[:, :-1] 
y = data.iloc[:, -1]

# Handle categories if any exist
x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

joblib.dump(model, "diabetes.pkl")
print("Success! Diabetes model saved as diabetes.pkl")