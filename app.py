import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Vehicle Trajectory Prediction", layout="centered")

st.title("🚗 Vehicle Trajectory Prediction System")
st.markdown("### AI-Based Future Position Prediction")

# ---------------------------
# Generate Training Data
# ---------------------------
np.random.seed(42)

speed_data = np.random.uniform(10, 100, 300)
angle_data = np.random.uniform(5, 85, 300)
time_data = np.random.uniform(1, 10, 300)

g = 9.81
theta = np.radians(angle_data)

x_position = speed_data * np.cos(theta) * time_data
y_position = speed_data * np.sin(theta) * time_data - 0.5 * g * time_data**2

X = np.column_stack((speed_data, angle_data, time_data))
y = np.column_stack((x_position, y_position))

# ---------------------------
# Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
accuracy = r2_score(y_test, y_pred_test)

# ---------------------------
# User Input
# ---------------------------
st.sidebar.header("Enter Vehicle Parameters")

speed = st.sidebar.number_input("Initial Speed (m/s)", 0.0, 150.0, 50.0)
angle = st.sidebar.number_input("Movement Angle (degrees)", 0.0, 90.0, 45.0)
time = st.sidebar.number_input("Time (seconds)", 0.1, 20.0, 5.0)

if st.sidebar.button("Predict Future Position"):

    input_data = np.array([[speed, angle, time]])
    prediction = model.predict(input_data)

    predicted_x = prediction[0][0]
    predicted_y = prediction[0][1]

    st.subheader("📍 Predicted Future Position")
    st.success(f"Horizontal Distance (X): {predicted_x:.2f} meters")
    st.success(f"Vertical Position (Y): {predicted_y:.2f} meters")

    if predicted_y <= 0:
        st.warning("Vehicle has reached ground level.")

    st.markdown("---")
    st.info(f"Model Accuracy (R² Score): {accuracy:.4f}")

else:
    st.info("Enter parameters and click 'Predict Future Position'")