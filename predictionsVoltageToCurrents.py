voltages = [2.1, 3.9, -6.2, 7.9, 10.1, 12.2]
currents = [1, 2, 3, 4, 5, 6]

w_values = [0.45, 0.50, 0.55]
b_values = [-0.1, 0.0, 0.1]
best_w = None
best_b = None
best_error = float('inf')

#Flag any suspicious voltages
for v in voltages:
    if v < 0:
        print("Suspicious voltage:", v)
for w in w_values:
    for b in b_values:
        predicted_currents = [w * v + b for v in voltages]
        error = sum((p - c) ** 2 for p, c in zip(predicted_currents, currents))
        if error < best_error:
            best_error = error
            best_w = w
            best_b = b

print(f"Best w: {best_w}, Best b: {best_b}, Best error: {best_error}")
        