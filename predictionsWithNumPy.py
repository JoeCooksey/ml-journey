import numpy as np
# Data
voltages = np.array([2.1, 3.9, 6.2, 7.9, 10.1, 12.2])
currents = np.array([1, 2, 3, 4, 5, 6])
# Parameters to search
w_values = [0.45, 0.50, 0.55]
b_values = [-0.1, 0.0, 0.1]

best_mse = float("inf")
best_w = None
best_b = None

for w in w_values:
    for b in b_values:
        # Model Prediction
        predicted_currents = w * voltages + b
        # Loss Function
        mse = np.mean((predicted_currents - currents) ** 2)
        # Best Parameters update
        if mse < best_mse:
            best_mse = mse
            best_w = w
            best_b = b

print("Best MSE:", best_mse)
print("Best w:", best_w)
print("Best b:", best_b)
