import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Autonomous Driving Assistance", layout="centered")

st.title("🚗 Autonomous Vehicle Driving Assistance System")
st.markdown("AI-based future position & collision prediction")

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
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Vehicle Sensor Inputs")

speed_input = st.sidebar.number_input("Current Speed (km/h)", 0.0, 200.0, 60.0)
acc_input = st.sidebar.number_input("Acceleration (m/s²)", -10.0, 10.0, 0.0)
steer_input = st.sidebar.number_input("Steering Angle (degrees)", -45.0, 45.0, 0.0)
dist_input = st.sidebar.number_input("Distance to Front Vehicle (meters)", 0.0, 200.0, 30.0)

if st.sidebar.button("Predict Future Movement"):

    input_data = np.array([[speed_input, acc_input, steer_input, dist_input]])
    predicted_position = model.predict(input_data)[0]

    st.subheader("🔮 Predicted Future Movement")
    st.success(f"Estimated Forward Movement: {predicted_position:.2f} meters in next 2 seconds")

    if dist_input < 10:
        st.error("⚠️ Collision Risk Detected! Reduce Speed Immediately.")
    elif dist_input < 20:
        st.warning("⚠️ Maintain Safe Distance.")
    else:
        st.success("✅ Safe Distance Maintained.")

else:
    st.info("Enter vehicle sensor values and click Predict.")