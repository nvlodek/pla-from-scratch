# Perceptron Learning Algorithm (PLA)
**UMSL CMP SCI 4300/5300 | Matt Vlodek & Noah Vlodek | September 2024**

## Overview
This project implements the Perceptron Learning Algorithm (PLA) from scratch in Python. The algorithm is tested on three datasets — linearly separable training data, non-linearly separable training data, and held-out test data — with experiments varying initial weights, learning rate, and point ordering.

---

## Features
- Custom `PLA` class built with NumPy (no ML libraries)
- Supports configurable learning rate, max iterations, and stagnation tolerance
- Tested on both linearly and non-linearly separable data
- Experiments with varied hyperparameters (weights, step size, point ordering)
- Misclassification error tracked on training and test sets

---

## Results Summary

### Linearly Separable Data
| Metric | Value |
|---|---|
| Weight vector updates | 8 |
| Iterations over training set | 4 |
| Training misclassification error | 0.0% |
| Test misclassification error | 10.0% |

### Non-Linearly Separable Data
| Metric | Value |
|---|---|
| Weight vector updates | 1059 |
| Iterations over training set | 82 |
| Training misclassification error | 20.0% |
| Test misclassification error | 60.0% |

---

## Hyperparameter Experiments

The effect of varying the learning rate (η) was tested for both data cases. Key findings:
- **η = 0.5** produced the best result for linearly separable data (0% test error)
- Shuffling training point order improved test error from 10% → 0% (linear case)
- For non-linear data, no learning rate produced a true solution; stagnation detection halted training

---

## Project Structure
```
├── pla.py              # PLA class implementation + data generation
├── report.pdf          # Full written report
└── README.md
```

---

## Requirements
```
pip install numpy pandas matplotlib scipy
```

---

## Usage
```python
from pla import PLA

model = PLA(learning_rate=1.0, max_iters=1000)
model.fit(X_train, y_train)
print(model.get_weights())
print(model.get_weight_updates())
print(model.get_iters())
```

---

## Dataset
All data is synthetically generated within the script using NumPy random functions. No external dataset is required.
