"""
Step 4: A proper 1D Kalman Filter.

The complementary filter uses a FIXED weighting (alpha=0.98) chosen by hand.
The Kalman filter instead computes the OPTIMAL weighting at every step,
based on how uncertain the current estimate is vs how noisy each sensor is.
This makes it more principled and adaptive.

State: angle (theta)
Process model: theta_k = theta_(k-1) + gyro_rate * dt   (prediction)
Measurement model: accel_angle is a noisy observation of theta   (update/correction)

Two steps per timestep:
  1. PREDICT: use the gyro to guess the new angle, and grow our uncertainty
     (since gyro integration accumulates error over time).
  2. UPDATE: use the accelerometer reading to correct that guess, weighted
     by the "Kalman gain" - how much we trust the measurement vs our
     current prediction uncertainty.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("/home/claude/imu_fusion/simulated_data.npz")
t, true_angle, accel_angle, gyro_rate, dt = (
    data["t"], data["true_angle"], data["accel_angle"], data["gyro_rate"], data["dt"]
)

# --- Kalman filter tuning parameters ---
# These represent how much we trust each sensor.
# In a real project, you'd measure these from real sensor datasheets/data
# (e.g. by computing the variance of a stationary sensor, as in Project 2!).
process_noise_var = 0.001    # how much we trust the gyro-based prediction step
measurement_noise_var = 16.0  # how noisy the accelerometer is (variance = std^2 = 4^2)

# --- Kalman filter state ---
angle_est = 0.0      # our best estimate of the angle
p_est = 1.0          # our uncertainty (variance) in that estimate

kalman_angle = np.zeros_like(t)

for i in range(len(t)):
    # ---- PREDICT step ----
    # Use gyro to predict the new angle
    angle_pred = angle_est + gyro_rate[i] * dt
    # Our uncertainty grows because we're relying on an imperfect prediction
    p_pred = p_est + process_noise_var

    # ---- UPDATE step ----
    # Kalman gain: how much to trust the new accelerometer measurement
    # vs our prediction. If p_pred is large (we're unsure), gain is high
    # (trust the measurement more). If measurement_noise_var is large
    # (sensor is noisy), gain is low (trust our prediction more).
    kalman_gain = p_pred / (p_pred + measurement_noise_var)

    # Correct the prediction using the accelerometer measurement
    angle_est = angle_pred + kalman_gain * (accel_angle[i] - angle_pred)
    # Update our uncertainty - it shrinks because we just incorporated new info
    p_est = (1 - kalman_gain) * p_pred

    kalman_angle[i] = angle_est

# --- Plot: compare all three methods together ---
plt.figure(figsize=(10, 5))
plt.plot(t, true_angle, "k-", linewidth=2, label="True angle")
plt.plot(t, accel_angle, "r.", markersize=2, alpha=0.25, label="Accelerometer (noisy)")
plt.plot(t, kalman_angle, "m-", linewidth=2, label="Kalman filter")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Kalman filter: optimally-weighted sensor fusion")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/imu_fusion/04_kalman_filter.png", dpi=120)
print("Saved plot: 04_kalman_filter.png")

rmse_kalman = np.sqrt(np.mean((kalman_angle - true_angle) ** 2))
rmse_accel_only = np.sqrt(np.mean((accel_angle - true_angle) ** 2))
print(f"\nRMSE - accelerometer only: {rmse_accel_only:.2f} degrees")
print(f"RMSE - Kalman filter: {rmse_kalman:.2f} degrees")
print(f"Improvement: {(1 - rmse_kalman/rmse_accel_only)*100:.0f}% lower error than raw accelerometer")
