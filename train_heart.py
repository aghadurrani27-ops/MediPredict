import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Data - Ensure path has the 's' in datasets
data = pd.read_csv("datasets/heart/heart.csv")

# 2. Use iloc to separate data
# x = all columns except the last one
# y = only the last column (the target)
x = data.iloc[:, :-1] 
y = data.iloc[:, -1]

# 3. Handle Categorical Data
# This converts any text columns into numbers so the model can read them
x = pd.get_dummies(x)

# 4. Split Data (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# 6. Save Model
joblib.dump(model, "heart.pkl")

print("Success! Heart model trained and saved as heart.pkl")
print(f"Columns used for training: {list(x.columns)}")
