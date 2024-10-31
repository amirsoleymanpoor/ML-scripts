#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

# Load Excel file
file_path = 'file:///C:/Users/test/Desktop/TGJO.xlsx'  # Enter your Excel file path
df = pd.read_excel(file_path)

# Convert date column to datetime and set as index
date_column_name = 'Date-Gregorian'
df[date_column_name] = pd.to_datetime(df[date_column_name])
df.set_index(date_column_name, inplace=True)

# Calculate returns
currency_column_name = 'Close'
df['Return'] = df[currency_column_name].pct_change() * 100
df['Lagged_Return'] = df['Return'].shift()
df = df.dropna()

# Slice the DataFrame
train = df['2024-09-25':'2024-10-26']
last_date = df.index[-1]
test = df[last_date:last_date]

# Create training and testing sets
X_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
X_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]

# Ask user for current gold price
current_gold_price_input = input("Please enter the current gold price in IRR (without commas): ")
current_gold_price_irr = float(current_gold_price_input.replace(',', ''))

# Train and evaluate Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Calculate RMSE and MAE for Random Forest
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f"Random Forest Prediction: {rf_predictions[0]:.2f}")
print(f"Actual Value: {y_test.values[0]:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

# Calculate RMSE and MAE for optimized model
best_rf_rmse = np.sqrt(mean_squared_error(y_test, best_rf_predictions))
best_rf_mae = mean_absolute_error(y_test, best_rf_predictions)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Optimized Random Forest Prediction: {best_rf_predictions[0]:.2f}")
print(f"Optimized Random Forest RMSE: {best_rf_rmse:.2f}")
print(f"Optimized Random Forest MAE: {best_rf_mae:.2f}")

# Calculate predicted gold price in IRR
predicted_return = rf_predictions[0]  # Predicted return
predicted_gold_price_irr = current_gold_price_irr * (1 + predicted_return / 100)

print(f"Predicted Gold Price in IRR: {predicted_gold_price_irr:.2f} IRR")
print(f"Current Gold Price in IRR: {current_gold_price_irr:.2f} IRR")

# Wait for user input before closing
input("Press Enter to exit...")


# In[ ]:




