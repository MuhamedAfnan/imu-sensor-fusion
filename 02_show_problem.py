"""
Step 2: Demonstrate WHY you need sensor fusion.

If you only use the accelerometer: your angle estimate is noisy every sample.
If you only integrate the gyroscope: small errors accumulate over time and
your estimate drifts away from the truth, even though each reading looked
smooth and reasonable.

This script visualizes both failure modes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt

data = np.load("/home/claude/imu_fusion/simulated_data.npz")
t, true_angle, accel_angle, gyro_rate, dt = (
    data["t"], data["true_angle"], data["accel_angle"], data["gyro_rate"], data["dt"]
)

# --- Naive approach 1: trust accelerometer only ---
# (just the raw noisy accel_angle itself)

# --- Naive approach 2: trust gyroscope only (integrate the rate) ---
gyro_only_angle = np.zeros_like(t)
angle = 0.0
for i in range(len(t)):
    angle += gyro_rate[i] * dt
    gyro_only_angle[i] = angle

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, true_angle, "k-", label="True angle", linewidth=2)
axes[0].plot(t, accel_angle, "r.", markersize=2, alpha=0.5, label="Accelerometer only (noisy)")
axes[0].set_ylabel("Angle (degrees)")
axes[0].set_title("Accelerometer alone: noisy but does not drift")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(t, true_angle, "k-", label="True angle", linewidth=2)
axes[1].plot(t, gyro_only_angle, "b-", label="Gyroscope only (integrated)")
axes[1].set_ylabel("Angle (degrees)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Gyroscope alone: smooth but DRIFTS away from truth over time")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/imu_fusion/02_the_problem.png", dpi=120)
print("Saved plot: 02_the_problem.png")

# Quantify the drift
final_error = abs(gyro_only_angle[-1] - true_angle[-1])
print(f"\nGyro-only final error after {t[-1]:.0f}s: {final_error:.1f} degrees of drift")
print("This is exactly why gyro-only dead-reckoning fails over time,")
print("and why accelerometer correction is needed to keep it anchored to truth.")
