"""
House Price Prediction v2 — Deployment App

Loads the trained pipeline (house_price_model.pkl) and lets users
input house features to get a live price prediction.

Run with: streamlit run app.py
"""

# ---------- STEP 0: Import Libraries ----------
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- STEP 1: Page Setup ----------
st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California House Price Predictor")
st.markdown(
    "Enter house details below to get a predicted price, using an "
    "**XGBoost model** trained on 20,640 real California housing records "
    "(1990 census data)."
)

# ---------- STEP 2: Load the Trained Pipeline ----------
# @st.cache_resource keeps the model loaded in memory across user
# interactions, instead of reloading the .pkl file on every prediction.
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# ---------- STEP 3: User Inputs ----------
st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    longitude = st.slider("Longitude", -124.5, -114.0, -119.5, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 36.5, 0.1)
    housing_median_age = st.slider("House Age (years)", 1, 52, 20)
    total_rooms = st.number_input("Total Rooms (in the block)", 1, 40000, 2500)
    total_bedrooms = st.number_input("Total Bedrooms (in the block)", 1, 7000, 500)
    population = st.number_input("Population (in the block)", 1, 40000, 1400)

with col2:
    households = st.number_input("Households (in the block)", 1, 6100, 500)
    median_income = st.slider("Median Income (tens of thousands $)", 0.5, 15.0, 3.8, 0.1)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

# ---------- STEP 4: Feature Engineering (must match training exactly) ----------
# These derived features must be computed the SAME way as in training,
# otherwise the model would receive inconsistent input.
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# ---------- STEP 5: Prediction ----------
if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
    }])

    # Model was trained on log(price), so we reverse the transform
    # (expm1 = inverse of log1p) to get the actual dollar prediction.
    log_prediction = model.predict(input_df)[0]
    prediction = np.expm1(log_prediction)

    st.success(f"### Predicted House Value: ${prediction:,.0f}")
    st.caption(
        "This is a statistical estimate based on 1990 census patterns, "
        "not a real-time market valuation."
    )

# ---------- STEP 6: Model Info ----------
st.divider()
st.caption(
    "Model: XGBoost Regressor | Trained on California Housing Dataset (20,640 records) | "
    "Test R² = 0.83 | Test MAPE = 15.8%"
)
