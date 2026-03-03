import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vehicle Trajectory Prediction", layout="wide")

st.title("🚗 Vehicle Trajectory Prediction System")

st.markdown("### Enter Vehicle Parameters")

# Sidebar inputs
st.sidebar.header("Input Parameters")

speed = st.sidebar.slider("Initial Speed (m/s)", 0, 100, 20)
angle = st.sidebar.slider("Launch Angle (degrees)", 0, 90, 45)
time = st.sidebar.slider("Time Duration (seconds)", 1, 20, 10)

# Convert angle to radians
theta = np.radians(angle)

# Physics trajectory formula
t = np.linspace(0, time, 100)
g = 9.81

x = speed * np.cos(theta) * t
y = speed * np.sin(theta) * t - 0.5 * g * t**2

# Remove negative height values
y = np.maximum(y, 0)

st.markdown("### 📊 Predicted Trajectory")

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.set_title("Vehicle Trajectory Path")
ax.grid()

st.pyplot(fig)

st.success("Prediction Generated Successfully ✅")