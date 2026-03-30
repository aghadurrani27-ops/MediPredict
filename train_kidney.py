import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("datasets/kidney/kidney.csv")

# Clean missing values (Kidney data often has empty cells)
data = data.dropna()

# Auto-separate: Features (x) and Target (y)
x = data.iloc[:, :-1] 
y = data.iloc[:, -1]

# Convert categories (like 'rbc', 'pc') to numbers
x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

joblib.dump(model, "kidney.pkl")
print("Success! Kidney model saved as kidney.pkl")