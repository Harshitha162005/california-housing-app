# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and metrics
model = joblib.load("regression_model.pkl")
metrics = joblib.load("metrics.pkl")

st.title("🏠 California Housing Price Predictor")
st.write("Enter the values below to predict the median house value (in 100,000s USD):")

# Input fields for features
MedInc = st.number_input("Median Income", min_value=0.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-124.0, value=-118.0)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]],
                              columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                                       "Population", "AveOccup", "Latitude", "Longitude"])
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Predicted Median House Value: ${prediction * 100000:,.2f}")

st.markdown("---")
st.header("📊 Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {metrics['MAE']:.2f}")
st.write(f"**Mean Squared Error (MSE):** {metrics['MSE']:.2f}")
st.write(f"**R² Score:** {metrics['R2']:.2f}")
