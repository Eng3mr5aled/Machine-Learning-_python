import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('advertising.csv')
print('Shape:', df.shape)

df.head(20)

# Exploratory scatter plots
plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

plt.scatter(df['Radio'], df['Sales'])
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show()

plt.scatter(df['Newspaper'], df['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.show()

# Simple Linear Regression (TV -> Sales)
x = df[['TV']]
y = df['Sales']

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'Train: {len(x_tr)} rows  |  Test: {len(x_te)} rows')

model1 = LinearRegression()
model1.fit(x_tr, y_tr)
print(f'w0 = {model1.intercept_}')  # intercept
print(f'w1 = {model1.coef_[0]}')    # coef1

y_pred1 = model1.predict(x_te)
Rmse = np.sqrt(mean_squared_error(y_te, y_pred1))
r2   = r2_score(y_te, y_pred1)
r2   = model1.score(x_te, y_te)
print(f'RMSE : ${Rmse:,.0f}')
print(f'R²   : {r2:.4f}')

plt.scatter(x_te, y_te, color='steelblue', label='Actual')
plt.plot(x_te, y_pred1, color='tomato', lw=2, label='Regression line')
plt.xlabel('TV')
plt.ylabel('Sales ($)')
plt.title('Simple LR — sklearn')
plt.legend()
plt.show()

# Multiple Linear Regression (TV, Radio, Newspaper -> Sales)
x_multi = df[['TV', 'Radio', 'Newspaper']]
y_multi = df['Sales']

x_tr_m, x_te_m, y_tr_m, y_te_m = train_test_split(x_multi, y_multi, test_size=0.2, random_state=42)
print(f'Train: {len(x_tr_m)} rows  |  Test: {len(x_te_m)} rows')

model = LinearRegression()
model.fit(x_tr_m, y_tr_m)
print(f'w0 = {model.intercept_}')  # intercept
print(f'w1 = {model.coef_[0]}')    # coef1
print(f'w2 = {model.coef_[1]}')    # coef2
print(f'w3 = {model.coef_[2]}')    # coef3

y_pred = model.predict(x_te_m)
rmse = np.sqrt(mean_squared_error(y_te_m, y_pred))
r2_m = r2_score(y_te_m, y_pred)
print(f'RMSE : ${rmse:,.0f}')
print(f'R²   : {r2_m:.4f}')
print(f'Simple LR R²: {r2:.4f}  →  Multiple LR R²: {r2_m:.4f}')

plt.scatter(y_te_m, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()
