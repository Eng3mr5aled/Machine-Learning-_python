# Advertising Sales Prediction — Linear Regression

A simple, from-scratch walkthrough of **Simple** and **Multiple Linear Regression** using `scikit-learn`, applied to the classic Advertising dataset. The goal is to predict product `Sales` based on advertising spend across three channels: **TV**, **Radio**, and **Newspaper**.

## 📊 Dataset

`advertising.csv` contains 200 records with the following columns:

| Column      | Description                                        |
|-------------|-----------------------------------------------------|
| `TV`        | Advertising budget spent on TV (in $1000s)          |
| `Radio`     | Advertising budget spent on Radio (in $1000s)        |
| `Newspaper` | Advertising budget spent on Newspaper (in $1000s)    |
| `Sales`     | Product sales (in thousands of units)                |

## 📓 Notebook Overview

`advertising_LR.ipynb` walks through the following steps:

1. **Setup** — import `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. **Load Data** — read `advertising.csv` into a DataFrame and inspect its shape.
3. **Exploratory Plots** — scatter plots of `Sales` against each advertising channel (`TV`, `Radio`, `Newspaper`) to visually inspect linear relationships.
4. **Part 1 — Simple Linear Regression**
   - Predicts `Sales` using only `TV` spend (the feature with the clearest linear trend).
   - Splits data 80/20 into train/test sets.
   - Fits a `LinearRegression` model and reports intercept/coefficient.
   - Evaluates performance with **RMSE** and **R²**.
   - Plots the fitted regression line against actual test data.
5. **Part 2 — Multiple Linear Regression**
   - Predicts `Sales` using all three features (`TV`, `Radio`, `Newspaper`).
   - Same train/test split, fitting, and evaluation process.
   - Compares R² of the multiple regression model against the simple model.
   - Plots actual vs. predicted sales.

## 📈 Results

| Model                       | R² Score |
|------------------------------|----------|
| Simple LR (`TV` only)        | ~0.80    |
| Multiple LR (all features)   | ~0.91    |

Using all three advertising channels together noticeably improves predictive performance over using `TV` spend alone.

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 Usage

1. Clone this repository.
2. Make sure `advertising.csv` is in the same directory as the notebook.
3. Open and run `advertising_LR.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

```bash
jupyter notebook advertising_LR.ipynb
```

## 📁 Repository Structure

```
.
├── advertising.csv        # Dataset
├── advertising_LR.ipynb   # Analysis & modeling notebook
└── README.md               # Project documentation
```

## 📝 License

Feel free to use and adapt this project for learning purposes.

## ✍️ Author

**Amr Khaled Sedik**
Computer Engineering, Ain Shams University
GitHub: [@Eng3mr5aled](https://github.com/Eng3mr5aled)
