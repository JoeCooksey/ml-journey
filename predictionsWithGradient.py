import numpy as np

voltages = np.array([2.1, 3.9, 6.2, 7.9, 10.1, 12.2])
currents = np.array([1, 2, 3, 4, 5, 6])

def mse_for_w_b(w, b):
    predicted_currents = w * voltages + b
    return np.mean((predicted_currents - currents) ** 2)

w = 0.40
b = 0.01
small_step = 0.001
learning_rate = 0.01

for i in range(1000):
    predicted_currents = w * voltages + b
    errors = predicted_currents - currents

    gradient_w = 2 * np.mean(errors * voltages)
    gradient_b = 2 * np.mean(errors)

    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    if i % 100 == 0:
        print(f"Step {i}, MSE: {np.mean(errors ** 2)}, w: {w}, b: {b}")

print("Final w:", w)
print("Final b:", b)

import matplotlib.pyplot as plt

predicted_currents = w * voltages + b

plt.scatter(voltages, currents, label="Measured data")
plt.plot(voltages, predicted_currents, label="Model prediction")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.legend()
plt.show()