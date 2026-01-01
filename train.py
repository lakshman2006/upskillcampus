import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load encoded dataset
df = pd.read_csv("EDA_ENCODED_DATASET.csv")

# 🔧 FIX 1: standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# 🔧 FIX 2: create ONE production target from year-wise columns
production_cols = [c for c in df.columns if c.startswith("production_")]

df["production"] = df[production_cols].mean(axis=1)

# (optional but clean) drop year-wise production columns
df = df.drop(columns=production_cols)

# 🔧 FIX 3: correct target name
target = "production"

# Derived feature (unchanged)
if "cost" in df.columns and "quantity" in df.columns:
    df["cost_per_quantity"] = df["cost"] / (df["quantity"] + 1)

# Feature / target split (unchanged)
X = df.drop(columns=[target])
y = df[target]

# Scale numeric features (unchanged)
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# Train-test split (unchanged)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-test split completed successfully.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- RANDOM FOREST MODEL ----------

# Initialize the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)


# ---------- MODEL EVALUATION ----------

# Predict on test data
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print("\nRandom Forest Model Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

import joblib

joblib.dump(rf_model, "crop_production_random_forest_model.pkl")

print("Final Random Forest model saved successfully.")
