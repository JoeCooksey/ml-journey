import numpy as np

X = np.array([
    [25,  6],
    [25,  8],
    [25, 10],
    [50,  6],
    [50,  8],
    [50, 10],
    [75,  6],
    [75,  8],
    [75, 10]
], dtype=float)

actual = np.array([
    17, 11, 5,
    22, 16, 10,
    27, 21, 15
], dtype=float)

weights = np.array([0.0, 0.0])
b = 0.0
learning_rate = 0.0001
for i in range(1000):
    predicted = X @ weights + b
    errors = predicted - actual

    gradient_weights = 2 * X.T @ errors / len(actual)
    gradient_b = 2 * np.mean(errors)

    weights -= learning_rate * gradient_weights
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print(f"Step {i}, MSE: {np.mean(errors ** 2)}, weights: {weights}, b: {b}")

print("Final weights:", weights)
print("Final b:", b)
print("Final RMSE:", np.sqrt(np.mean(errors ** 2)))