import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit page
st.set_page_config(page_title="⚡ Electricity Bill Predictor", page_icon="⚡", layout="centered")

st.markdown("<h1 style='text-align: center; color:#4B8BBE;'>⚡ Smart Electricity Bill Predictor</h1>", unsafe_allow_html=True)
st.caption("📊 Predict your daily electricity usage and bill using a trained ML model (Linear Regression).")

# Load data
@st.cache_data
def load_data():
   return pd.read_csv("electricity_usage_dataset.csv")

data = load_data()

# Define features and target
features = ['Fans', 'Lights', 'ACs', 'Refrigerator', 'Water_Pump', 'Temperature', 'Time']
X = data[features]
y = data[['Consumption_kWh']]

# Train model
model = LinearRegression()
model.fit(X, y)
predicted = model.predict(X)
mse = mean_squared_error(y, predicted)
r2 = r2_score(y, predicted)

# Sidebar inputs
st.sidebar.header("🔌 Enter Usage Details")
fans = st.sidebar.slider("🌀 Fans Running", 0, 10, 2)
lights = st.sidebar.slider("💡 Lights On", 0, 20, 5)
acs = st.sidebar.slider("❄️ ACs On", 0, 3, 1)
fridge = st.sidebar.radio("🧊 Refrigerator", ["On", "Off"])
pump = st.sidebar.radio("🚿 Water Pump", ["On", "Off"])
temp = st.sidebar.slider("🌡️ Temperature (°C)", 15, 45, 30)
time = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)

# Optional rate selection
rate_per_unit = st.sidebar.selectbox("💰 Rate per Unit (kWh)", [45, 50, 55, 60], index=2)

# Prepare input for prediction
fridge_val = 1 if fridge == "On" else 0
pump_val = 1 if pump == "On" else 0
input_data = np.array([[fans, lights, acs, fridge_val, pump_val, temp, time]])

# Prediction section
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(input_data)[0][0]
    bill = prediction * rate_per_unit
    monthly = bill * 30

    st.markdown("## 🔎 Prediction Result")
    st.success(f"**Estimated Daily Consumption:** `{prediction:.2f} kWh`")
    st.info(f"**Estimated Daily Bill:** `Rs. {bill:.2f}`")
    st.markdown(f"💸 **Estimated Monthly Bill:** `Rs. {monthly:.0f}`")
    st.warning(f"📉 Model Mean Squared Error (MSE): `{mse:.3f}` | R² Score: `{r2:.3f}`")

    # Save to session history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Fans": fans, "Lights": lights, "ACs": acs, "Refrigerator": fridge_val,
        "Water_Pump": pump_val, "Temp": temp, "Time": time,
        "kWh": round(prediction, 2), "Bill": round(bill, 2)
    })

# View prediction history
if 'history' in st.session_state and len(st.session_state.history) > 0:
    st.markdown("### 🧾 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

# Feature importance (absolute coefficients)
st.markdown("### 📊 Feature Influence on Consumption")
coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=coeff_df, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
ax.set_title("Feature Importance (Linear Coefficients)")
st.pyplot(fig)

st.markdown("---")
st.caption("🧠 Developed with Streamlit & Scikit-Learn | By **Kashi-Ch** — A Practical Machine Learning Project")



