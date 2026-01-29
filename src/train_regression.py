# src/train_regression.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import the correct preprocessing function
from src.data_preprocessing import preprocess_data

# -----------------------------
# Step 1: Load and preprocess data
# -----------------------------
df = preprocess_data()  # Returns the processed DataFrame

# -----------------------------
# Step 2: Define features and target
# -----------------------------
# Drop columns that are not features (target + any non-numeric columns)
# Based on your CSV columns, we drop 'price' (target) and 'price 2' (optional)
if 'price' in df.columns and 'price 2' in df.columns:
    X = df.drop(columns=['price', 'price 2'])
    y = df['price']  # Target variable
else:
    raise KeyError("Columns 'price' or 'price 2' not found in DataFrame")

# Convert categorical columns to numeric if needed
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Step 3: Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train regression model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Regression Model Trained")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------
# Step 6: Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/regression_model.pkl")
print("✅ Model saved at: models/regression_model.pkl")

