# ML Journey

Learning machine learning from first principles — no scikit-learn, no autograd, just
NumPy and the math. Each script is a step up in complexity, building toward a complete
model-selection workflow for a real electronics problem: predicting a power MOSFET's
on-resistance (Rds(on)) from operating conditions.

The goal isn't to use ML libraries — it's to understand what they do under the hood by
implementing linear regression, gradient descent, regularization, and model selection by
hand.

## The progression

| # | Script | Concept |
|---|--------|---------|
| 1 | [`predictionsCurrentToVoltages.py`](predictionsCurrentToVoltages.py) | Manual grid search over a single bias term using plain Python lists and sum-of-squared-errors. |
| 2 | [`predictionsVoltageToCurrents.py`](predictionsVoltageToCurrents.py) | Grid search over both slope `w` and bias `b`, plus basic input validation (flagging physically suspicious values). |
| 3 | [`predictionsWithNumPy.py`](predictionsWithNumPy.py) | Same grid search, vectorized with NumPy — replacing explicit loops with array math. |
| 4 | [`predictionsWithGradient.py`](predictionsWithGradient.py) | Gradient descent replaces brute-force search — analytic gradients for `w` and `b`, with a fitted-line plot. |
| 5 | [`predictionsQuadratic.py`](predictionsQuadratic.py) | Multi-feature linear regression in matrix form (`X @ weights + b`) trained by gradient descent. |
| 6 | [`mosfetPredictor.py`](mosfetPredictor.py) | The full workflow — see below. |

## The capstone: `mosfetPredictor.py`

A complete, honest model-selection pipeline for predicting MOSFET Rds(on) (in mΩ) from
temperature, gate voltage, and drain current:

- **Train / validation / test split** (70 / 15 / 15) with feature standardization, using
  *training-set* statistics only to avoid data leakage.
- **L2 regularization** (ridge) implemented by hand, with the penalty added directly to
  the weight gradient.
- **Hyperparameter tuning** — sweeps λ over `[0.0, 0.001, 0.01, 0.1, 0.5, 1.0]` and
  selects the value that minimizes validation RMSE (λ = 0.01).
- **Model comparison** — a 5-feature model (including two correlated temperature sensors)
  vs. a simpler 3-feature model. The redundant sensors add nothing, so the simpler model
  wins. Final **test RMSE ≈ 0.76 mΩ**, evaluated once on the untouched test set.
- **Error analysis** — isolates the worst under-prediction (the dangerous direction for a
  thermal margin) using *signed* residuals, and plots residuals vs. temperature to check
  for systematic bias / multicollinearity.

**Main lesson:** the simplest model that performs just as well is the better engineering
model.

### Next step

Add a `temperature²` feature to capture curvature — giving four inputs (temperature,
temperature², gate voltage, drain current) — and test whether modeling the nonlinearity
beats the plain linear fit.

## Running

```bash
pip install numpy matplotlib
python mosfetPredictor.py
```

> Note: the synthetic data uses a fixed RNG seed, so results are reproducible. Plotting
> blocks are present (some commented out) and require a display.
