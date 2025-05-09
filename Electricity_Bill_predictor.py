import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="⚡", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚡ Smart Electricity Bill Predictor</h1>", unsafe_allow_html=True)
st.caption("📊 Predict daily electricity consumption & billing using a trained Machine Learning model (Linear Regression).")
@st.cache_data
def load_data():
    return  pd.read_csv("electricity_usage_dataset.csv")

data = load_data()
x = data[['Fans','Lights','ACs','Refrigerator','Water_Pump','Temperature','Time',]]
y = data[['Consumption_kWh']]


model = LinearRegression()
model.fit(x, y)
mse = mean_squared_error(y, model.predict(x))

st.sidebar.header("🔌 Enter Your Usage Details")
st.sidebar.markdown("Please provide the following details to predict your electricity consumption and bill.")
fans = st.sidebar.slider("🌀 Fans Running", 0, 10, 2, help="Number of ceiling or pedestal fans currently in use.")
lights = st.sidebar.slider("💡 Lights On", 0, 20, 5, help="Total lights turned on in your home.")
acs = st.sidebar.slider("❄️ ACs On", 0, 3, 1, help="Air Conditioners currently running.")
fridge = st.sidebar.radio("🧊 Refrigerator", ["On", "Off"])
pump = st.sidebar.radio("🚿 Water Pump", ["On", "Off"])
temp = st.sidebar.slider("🌡️ Temperature (°C)", 15, 45, 30, help="Current outside temperature.")
time = st.sidebar.slider("⏰ Time of Day (Hour)", 0, 23, 14, help="Current hour in 24-hour format.")


fridge_value = 1 if fridge == "On" else 0
pump_value = 1 if pump == "On" else 0

if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[fans, lights, acs, fridge_value, pump_value, temp, time]])
    predicted_kwh = model.predict(input_data)[0]
    rate_per_unit = 55
    estimated_bill = predicted_kwh * rate_per_unit


    st.markdown("## 🔎 Prediction Result")
    st.success(f"**Estimated Daily Electricity Consumption:** `{predicted_kwh[0]:.2f} kWh`")
    st.info(f"**Estimated Daily Bill:** `Rs. {estimated_bill[0]:.2f}`")
    
    st.warning(f"📉 Model Accuracy (MSE): `{mse:.3f}`")

    monthly_bill = estimated_bill * 30
    st.markdown(f"💸 **Estimated Monthly Bill:** `Rs. {monthly_bill.mean():.0f}`")

st.markdown("---")
st.caption("🧠 Built using Streamlit & Scikit-Learn | Developed by Kashi-Ch  — A practical ML project")


