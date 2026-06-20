import matplotlib.pyplot as plt
import numpy as np
# Generate random data
rng = np.random.default_rng(42)
n_samples = 500

temperature = rng.uniform(25, 125, n_samples)
gate_voltage = rng.uniform(6, 12, n_samples)
drain_current = rng.uniform(0.5, 20, n_samples)

ambient_temperature = temperature - rng.uniform(5, 15, n_samples)
redundant_temperature = temperature - rng.uniform(0, 1.5, n_samples)

noise = rng.normal(0, 0.8, n_samples)
# Rds_on Calculation
Rds_on = [
    20 
    + 0.12 * (temperature - 25)
    - 2.0 * (gate_voltage - 8)
    + 0.15 * drain_current
    + noise
]
# Graphs
x = np.column_stack([
    temperature,
    gate_voltage,
    drain_current,
    ambient_temperature,
    redundant_temperature
])

y = np.array(Rds_on).flatten()

features_names = [
    "temperature",
    "gate_voltage",
    "drain_current",
    "ambient_temperature",
    "redundant_temperature",
]

#Split data into training, validation, and test sets
indices = rng.permutation(n_samples)
train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

x_train, y_train = x[train_indices], y[train_indices]
x_val, y_val = x[val_indices], y[val_indices]
x_test, y_test = x[test_indices], y[test_indices]

train_mean = x_train.mean(axis=0)
train_std = x_train.std(axis=0)

X_train_scaled = (x_train - train_mean) / train_std
X_validation_scaled = (x_val - train_mean) / train_std
X_test_scaled = (x_test - train_mean) / train_std

# Add L2 Regularization
def train_l2_model(
    X_train_scaled,
    y_train,
    lambda_value,
    learning_rate=0.01,
    epochs=5000
):
    n_features = X_train_scaled.shape[1]

    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        predictions = X_train_scaled @ weights + bias
        errors = predictions - y_train

        gradient_weights = (
            2 * (X_train_scaled.T @ errors) / len(y_train)
            + 2 * lambda_value * weights
        )

        gradient_bias = 2 * np.mean(errors)

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias

# Train the L2-regularized model
lambda_values = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]

validation_rmses = []
trained_models = []

for lambda_value in lambda_values:
    weights, bias = train_l2_model(
        X_train_scaled,
        y_train,
        lambda_value
    )
    validation_predictions = (
        X_validation_scaled @ weights + bias
    )
    validation_rmse = (
        np.sqrt(
            np.mean((validation_predictions - y_val) ** 2)
        )
    )
    validation_rmses.append(validation_rmse)
    trained_models.append((weights, bias))

#Print results
for lambda_value, rmse in zip(lambda_values, validation_rmses):
    print(f"lambda={lambda_value:.3f}, validation RMSE={rmse:.4f}")

best_lambda = lambda_values[np.argmin(validation_rmses)]
print("Best lambda:", best_lambda)

#Now plot RMSE against lambda
"""
plt.figure()
plt.plot(lambda_values, validation_rmses, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Validation RMSE (mOhms)')
plt.title('Validation RMSE vs L2 Regularization')
plt.xscale("symlog", linthresh=0.001)
plt.grid(True)
plt.show()
"""

#Retrain final model with the best lambda
X_final_train = np.vstack([x_train, x_val])
y_final_train = np.concatenate([y_train, y_val])

final_mean = X_final_train.mean(axis=0)
final_std = X_final_train.std(axis=0)

X_final_train_scaled = (
    X_final_train - final_mean
) / final_std

X_test_scaled_final = (
    x_test - final_mean
) / final_std

final_weights, final_bias = train_l2_model(
    X_final_train_scaled,
    y_final_train,
    best_lambda
)

test_predictions = (
    X_test_scaled_final @ final_weights + final_bias
)

test_rmse = (
    np.sqrt(np.mean((test_predictions - y_test) ** 2))
)

test_rmse_5var = test_rmse

print("\n===== 5-VARIABLE MODEL =====")
print(f"Test RMSE: {test_rmse_5var:.4f}")
for name, weight in zip(features_names, final_weights):
    print(f"  {name}: {weight:.4f}")
print(f"  bias: {final_bias:.4f}")

# =============================================
# 3-VARIABLE MODEL (temperature, gate_voltage, drain_current)
# =============================================
core_cols = [0, 1, 2]
features_names_3 = [features_names[i] for i in core_cols]

x_train_3 = x_train[:, core_cols]
x_val_3 = x_val[:, core_cols]
x_test_3 = x_test[:, core_cols]

train_mean_3 = x_train_3.mean(axis=0)
train_std_3 = x_train_3.std(axis=0)

X_train_scaled_3 = (x_train_3 - train_mean_3) / train_std_3
X_val_scaled_3 = (x_val_3 - train_mean_3) / train_std_3
X_test_scaled_3 = (x_test_3 - train_mean_3) / train_std_3

validation_rmses_3 = []
trained_models_3 = []

for lambda_value in lambda_values:
    w3, b3 = train_l2_model(
        X_train_scaled_3, y_train, lambda_value
    )
    val_pred_3 = X_val_scaled_3 @ w3 + b3
    val_rmse_3 = np.sqrt(np.mean((val_pred_3 - y_val) ** 2))
    validation_rmses_3.append(val_rmse_3)
    trained_models_3.append((w3, b3))

best_lambda_3 = lambda_values[np.argmin(validation_rmses_3)]

X_final_train_3 = np.vstack([x_train_3, x_val_3])
y_final_train_3 = np.concatenate([y_train, y_val])

final_mean_3 = X_final_train_3.mean(axis=0)
final_std_3 = X_final_train_3.std(axis=0)

X_final_train_scaled_3 = (
    X_final_train_3 - final_mean_3
) / final_std_3
X_test_scaled_final_3 = (
    x_test_3 - final_mean_3
) / final_std_3

final_weights_3, final_bias_3 = train_l2_model(
    X_final_train_scaled_3, y_final_train_3, best_lambda_3
)

test_pred_3 = X_test_scaled_final_3 @ final_weights_3 + final_bias_3
test_rmse_3var = np.sqrt(np.mean((test_pred_3 - y_test) ** 2))

print("\n===== 3-VARIABLE MODEL =====")
print(f"Best lambda: {best_lambda_3}")
print(f"Test RMSE: {test_rmse_3var:.4f}")
for name, weight in zip(features_names_3, final_weights_3):
    print(f"  {name}: {weight:.4f}")
print(f"  bias: {final_bias_3:.4f}")

# =============================================
# COMPARISON
# =============================================
print("\n===== COMPARISON: 3 vs 5 VARIABLES =====")
print(f"  5-variable Test RMSE: {test_rmse_5var:.4f}")
print(f"  3-variable Test RMSE: {test_rmse_3var:.4f}")
diff = test_rmse_5var - test_rmse_3var
print(f"  Difference (5var - 3var): {diff:+.4f}")
if abs(diff) < 0.01:
    print("  -> Nearly identical — extra features add no value.")
elif diff > 0:
    print("  -> 3-variable model is better (simpler & more accurate).")
else:
    print("  -> 5-variable model is better (extra features help).")

"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, test_predictions, alpha=0.5, label="5-var")
axes[0].scatter(y_test, test_pred_3, alpha=0.5, label="3-var")
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "k--", label="Perfect")
axes[0].set_xlabel("Actual Rds_on (mΩ)")
axes[0].set_ylabel("Predicted Rds_on (mΩ)")
axes[0].set_title("Predicted vs Actual")
axes[0].legend()
axes[0].grid(True)

x_pos = np.arange(len(lambda_values))
width = 0.35
axes[1].bar(x_pos - width/2, validation_rmses, width, label="5-var")
axes[1].bar(x_pos + width/2, validation_rmses_3, width, label="3-var")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([str(l) for l in lambda_values])
axes[1].set_xlabel("Lambda")
axes[1].set_ylabel("Validation RMSE")
axes[1].set_title("Validation RMSE by Lambda")
axes[1].legend()
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.show()
"""
#Find the index of biggest under-prediction
index = np.argmin(test_pred_3 - y_test)

print("Inputs:", x_test_3[index])
print("Prediction:", test_pred_3[index])
print("Actual:", y_test[index])
print("Error:", test_pred_3[index] - y_test[index])

#plot residuals
errors = test_pred_3 - y_test

plt.figure()
plt.scatter(x_test_3[:, 0], errors)
plt.axhline(0)
plt.xlabel("Temperature (°C)")
plt.ylabel("Prediction error (mΩ)")
plt.title("Residuals vs Temperature")
plt.show()