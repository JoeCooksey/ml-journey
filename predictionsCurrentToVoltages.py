currents = [1, 2, 3, 4, 5, 6]
voltages = [2.1, 3.9, 6.2, 7.9, 10.1, 12.2]

w = 2
b_values = [0.00, 0.05, 0.10, 0.15]

min_error = float("inf")
best_b = None

for b in b_values:
    predictions = []

    for I in currents:
        predictions.append(w * I + b)

    squared_errors = []

    for i in range(len(predictions)):
        squared_errors.append((voltages[i] - predictions[i]) ** 2)

    total_error = sum(squared_errors)

    print("b:", b)
    print("predictions:", predictions)
    print("squared errors:", squared_errors)
    print("total error:", total_error)
    print()

    if total_error < min_error:
        min_error = total_error
        best_b = b

print(f"Minimum error: {min_error}, Best b value: {best_b}")
