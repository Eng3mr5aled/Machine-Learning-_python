# Student Performance Predictor

Machine learning pipeline for predicting student academic outcomes — final exam scores, pass/fail status, and letter grades — from study habits and attendance data. Built with scikit-learn, covering regression, binary and multi-class classification, and decision trees.

## 📊 Overview

This project explores a student performance dataset through three complementary ML tasks:

| Task | Type | Target | Model(s) |
|---|---|---|---|
| Final Score Prediction | Regression | `FinalScore_Target_Regression` | Linear Regression |
| Pass/Fail Prediction | Binary Classification | `Pass_Target_Logistic` | Logistic Regression |
| Grade Prediction | Multi-Class Classification | `Grade_Target_MultiClass` (A/B/C/Fail) | Logistic Regression, Decision Tree |

## 🧠 Features Used

- `StudyHours` — hours spent studying
- `Attendance` — attendance percentage
- `PrevExamScore` — score on a previous exam
- `SleepHours` — average hours of sleep

## 🔍 What's Inside

- **Exploratory analysis** — feature-vs-target scatter plots
- **Linear Regression** — predicts final score, evaluated with RMSE, MAE, R², plus a reusable `predict_score()` function
- **Logistic Regression (binary)** — predicts pass/fail with feature scaling, accuracy, and confusion matrix
- **Multi-class classification** — predicts letter grade (A/B/C/Fail) using:
  - Logistic Regression with softmax probabilities and cross-entropy loss
  - Decision Tree (entropy criterion) with tree structure printout and feature importance
- **Evaluation** — confusion matrices, classification reports, accuracy/precision/recall/F1 (macro-averaged)

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## 📁 Project Structure

```
├── students_project.ipynb      # Main notebook — full pipeline
├── student_performance.csv     # Dataset
└── README.md
```

## 🚀 Getting Started

1. Clone the repo
   ```bash
   git clone https://github.com/Eng3mr5aled/Student-Performance-Predictor.git
   cd Student-Performance-Predictor
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the notebook
   ```bash
   jupyter notebook students_project.ipynb
   ```

## 📈 Results Summary

- **Linear Regression:** evaluated via RMSE, MAE, and R² on held-out test data
- **Logistic Regression (binary):** accuracy on pass/fail classification
- **Multi-class models:** compared via classification report (precision, recall, F1 per grade) and confusion matrix; Decision Tree adds interpretable feature importances

## 👤 Author

**Amr Khaled Sedik**
Computer and Control Systems Engineering, Ain Shams University
GitHub: [Eng3mr5aled](https://github.com/Eng3mr5aled)
