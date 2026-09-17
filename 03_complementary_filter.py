"""
Step 3: The Complementary Filter.

The simplest possible sensor fusion. The idea:
- Trust the gyro-integrated angle in the SHORT term (it's smooth, low noise)
- But keep "pulling" the estimate back toward the accelerometer reading
  slowly, so long-term drift gets corrected.

This is a weighted average that leans on each sensor's strength:

    angle = alpha * (angle_prev + gyro_rate * dt) + (1 - alpha) * accel_angle

alpha close to 1 (e.g. 0.98) means: trust the gyro's short-term smoothness a
lot, but still let 2% of the accelerometer reading "correct" it every step.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("/home/claude/imu_fusion/simulated_data.npz")
t, true_angle, accel_angle, gyro_rate, dt = (
    data["t"], data["true_angle"], data["accel_angle"], data["gyro_rate"], data["dt"]
)

alpha = 0.98
comp_angle = np.zeros_like(t)
angle = 0.0
for i in range(len(t)):
    angle = alpha * (angle + gyro_rate[i] * dt) + (1 - alpha) * accel_angle[i]
    comp_angle[i] = angle

# --- Plot comparison ---
plt.figure(figsize=(10, 5))
plt.plot(t, true_angle, "k-", linewidth=2, label="True angle")
plt.plot(t, accel_angle, "r.", markersize=2, alpha=0.3, label="Accelerometer (noisy)")
plt.plot(t, comp_angle, "g-", linewidth=2, label=f"Complementary filter (alpha={alpha})")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Complementary filter: smooth AND anchored to truth (no drift)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/imu_fusion/03_complementary_filter.png", dpi=120)
print("Saved plot: 03_complementary_filter.png")

rmse = np.sqrt(np.mean((comp_angle - true_angle) ** 2))
rmse_accel_only = np.sqrt(np.mean((accel_angle - true_angle) ** 2))
print(f"\nRMSE - accelerometer only: {rmse_accel_only:.2f} degrees")
print(f"RMSE - complementary filter: {rmse:.2f} degrees")
print(f"Improvement: {(1 - rmse/rmse_accel_only)*100:.0f}% lower error than raw accelerometer")
