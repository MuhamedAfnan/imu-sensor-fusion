"""
Step 1: Simulate a realistic IMU scenario.

We create a "true" tilt angle that changes smoothly over time (like a sensor
slowly being tilted by hand), then generate two FAKE noisy sensor readings
that mimic how a real accelerometer and gyroscope behave:

- Accelerometer-derived angle: noisy on every sample, but does NOT drift
  over time (each reading is independent).
- Gyroscope (angular rate): smooth and less noisy per-sample, but integrating
  it over time causes the angle estimate to DRIFT away from the truth.

This is the exact problem that motivates sensor fusion.
"""

import numpy as np

np.random.seed(42)  # reproducible results

# --- Time setup ---
dt = 0.01          # 10 ms between samples (100 Hz, typical IMU rate)
t_end = 10.0        # 10 seconds of data
t = np.arange(0, t_end, dt)
n = len(t)

# --- "True" angle: a smooth motion, like slowly tilting a sensor by hand ---
true_angle = 20 * np.sin(0.5 * t) + 10 * np.sin(0.1 * t)  # degrees

# --- True angular rate (derivative of true angle) ---
true_rate = np.gradient(true_angle, dt)  # degrees/sec

# --- Simulate accelerometer-derived angle: noisy, no drift ---
accel_noise_std = 4.0  # degrees - accelerometers are noisy but don't drift
accel_angle = true_angle + np.random.normal(0, accel_noise_std, n)

# --- Simulate gyroscope reading: small noise + a slow bias that causes drift ---
gyro_noise_std = 0.5     # degrees/sec - gyros are smoother per-sample
gyro_bias = 0.3          # degrees/sec constant bias - this is what causes drift
gyro_rate = true_rate + gyro_bias + np.random.normal(0, gyro_noise_std, n)

# --- Save everything for the next steps ---
np.savez(
    "/home/claude/imu_fusion/simulated_data.npz",
    t=t, true_angle=true_angle, accel_angle=accel_angle, gyro_rate=gyro_rate, dt=dt
)

print(f"Simulated {n} samples over {t_end} seconds.")
print(f"Accelerometer noise std: {accel_noise_std} deg")
print(f"Gyro noise std: {gyro_noise_std} deg/s, bias: {gyro_bias} deg/s")
print("Saved to simulated_data.npz")
