# 📈 Linear Regression from Scratch — Predicting Restaurant Profits

A from-scratch implementation of univariate linear regression, trained with batch gradient descent to predict restaurant profits from city population data. Built as part of the Machine Learning Specialization (Course 1, Week 2) coursework, with the core cost and gradient functions implemented manually using only NumPy — no scikit-learn, no shortcuts.

---

## 🎯 Overview

Imagine you're the CEO of a restaurant franchise deciding which city to expand into next. You have historical data on city population and the corresponding monthly profit (or loss) of existing restaurants. The goal is to fit a linear model that can predict expected profit for a new city given only its population.

This project implements that pipeline end-to-end:
- Loading and visualizing the dataset
- Implementing the cost function $J(w,b)$ from first principles
- Implementing the gradient computation $\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$
- Running batch gradient descent to learn optimal parameters
- Visualizing the fitted line against the training data
- Using the learned model to predict profit for new populations

---

## 🧮 Model

The hypothesis is a simple linear function:

$$f_{w,b}(x) = wx + b$$

Trained by minimizing the mean squared error cost function:

$$J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$

using batch gradient descent, where both parameters are updated simultaneously on every iteration:

$$w := w - \alpha \frac{\partial J(w,b)}{\partial w}, \qquad b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$

---

## 📂 Repository Structure

| File | Description |
|---|---|
| `C1_W2_Linear_Regression.ipynb` | Main notebook — data exploration, cost/gradient implementation, training, visualization, and predictions |
| `utils.py` | Helper functions for loading the datasets into NumPy arrays |
| `data/ex1data1.txt` | Univariate dataset — city population vs. restaurant profit (97 examples) |
| `data/ex1data2.txt` | Bonus multivariate dataset — house size, bedroom count, and price (47 examples) |

> ⚠️ **Note:** `data/ex1data2.txt` is loaded via `utils.load_data_multi()` but isn't exercised in this notebook — it's included for a natural follow-up into multivariate linear regression.

---

## ⚙️ Setup

```bash
pip install numpy matplotlib
```

Make sure the `data/` folder (containing `ex1data1.txt` and `ex1data2.txt`) sits alongside `utils.py` in your working directory, then run the notebook cells in order.

---

## 🔑 Key Implementation Details

### 1. Compute Cost
Iterates over all training examples, accumulates squared error, and normalizes by $2m$:

```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = x[i] * w + b
        cost = (f_wb - y[i]) ** 2
        total_cost += cost
    return (1 / (2 * m)) * total_cost
```

### 2. Compute Gradient
Computes $\partial J/\partial w$ and $\partial J/\partial b$ by summing the per-example error terms:

```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw_sum, dj_db_sum = 0, 0
    for i in range(m):
        f_wb = x[i] * w + b
        error = f_wb - y[i]
        dj_dw_sum += error * x[i]
        dj_db_sum += error
    return dj_dw_sum / m, dj_db_sum / m
```

### 3. Batch Gradient Descent
Runs for a fixed number of iterations, updating $w$ and $b$ simultaneously each step while logging cost history for convergence checks.

---

## 📊 Results

| Metric | Value |
|---|---|
| Learning rate (α) | 0.01 |
| Iterations | 1,500 |
| Learned weight (w) | 1.16636 |
| Learned bias (b) | -3.63029 |
| Cost at initial (w=2, b=1) | 75.203 |

**Predictions on new cities:**

| City Population | Predicted Monthly Profit |
|---|---|
| 35,000 | $4,519.77 |
| 70,000 | $45,342.45 |

The fitted line closely tracks the upward trend in the scatter plot of profit vs. population, confirming gradient descent converged to a sensible minimum of the cost surface.

---

## 🧠 What This Demonstrates

- Manual implementation of the core mathematics behind linear regression (no black-box `.fit()` calls)
- Understanding of cost surfaces and how gradient descent navigates them
- Practical grounding in how learning rate and iteration count affect convergence
- A foundation directly extensible to multivariate regression (see `ex1data2.txt`) and regularization

---

## 🙏 Acknowledgments 
**Based on the Linear Regression practice lab from the *Machine Learning Specialization* (DeepLearning.AI / Stanford, taught by Andrew Ng), extended and documented independently.**
**Exercises done by Amr Khaled Sedik ,Computer Engineer, Ain Shams University**
