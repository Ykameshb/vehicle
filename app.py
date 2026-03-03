import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Autonomous Driving Assistance", layout="centered")

st.title("🚗 Autonomous Vehicle Driving Assistance System")
st.markdown("AI-based future movement & collision risk prediction")

st.markdown("---")

# ----------------------------
# Generate Synthetic Driving Data
# ----------------------------
np.random.seed(42)

speed = np.random.uniform(0, 120, 500)
acceleration = np.random.uniform(-5, 5, 500)
steering = np.random.uniform(-30, 30, 500)
distance_front = np.random.uniform(1, 100, 500)

future_position = speed * 2 + acceleration * 4 - steering * 0.5

X = np.column_stack((speed, acceleration, steering, distance_front))
y = future_position

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# User Inputs (Main Screen – Mobile Friendly)
# ----------------------------

st.subheader("Enter Vehicle Sensor Values")

speed_input = st.number_input("Current Speed (km/h)", 0.0, 200.0, 60.0)
acc_input = st.number_input("Acceleration (m/s²)", -10.0, 10.0, 0.0)
steer_input = st.number_input("Steering Angle (degrees)", -45.0, 45.0, 0.0)
dist_input = st.number_input("Distance to Front Vehicle (meters)", 0.0, 200.0, 30.0)

st.markdown("")

if st.button("🚀 Predict Future Movement"):

    input_data = np.array([[speed_input, acc_input, steer_input, dist_input]])
    predicted_position = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    st.success(f"Estimated Forward Movement: {predicted_position:.2f} meters (next 2 seconds)")

    # Collision Logic
    if dist_input < 10:
        st.error("⚠️ HIGH COLLISION RISK! Brake Immediately.")
    elif dist_input < 20:
        st.warning("⚠️ Maintain Safe Distance.")
    else:
        st.success("✅ Safe Distance Maintained.")

else:
    st.info("Fill sensor values and click Predict.")