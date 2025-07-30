# model.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model
joblib.dump(model, "regression_model.pkl")

# Save evaluation metrics
metrics = {
    "MAE": mae,
    "MSE": mse,
    "R2": r2
}
joblib.dump(metrics, "metrics.pkl")

print("Model trained and saved.")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")
