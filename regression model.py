#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Mock historical data for event industry revenue (in billions)
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Revenue': [1200, 1300, 1400, 1550, 1650, 900, 1100, 1400, 1500]
}

df = pd.DataFrame(data)

# Preparing data for regression
X = df[['Year']]
y = df['Revenue']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Forecast future revenue
future_years = pd.DataFrame({'Year': [2024, 2025, 2026, 2027]})
future_revenue = model.predict(future_years)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Revenue'], color='blue', label='Historical Data')
plt.plot(df['Year'], model.predict(X), color='red', label='Regression Line')
plt.scatter(future_years, future_revenue, color='green', label='Forecast')
plt.xlabel('Year')
plt.ylabel('Revenue (in billions)')
plt.title('Event Planning Industry Revenue Forecast')
plt.legend()
plt.show()

# Print Forecast Results
print("Forecasted Revenue:")
future_years['Forecasted Revenue (in billions)'] = future_revenue
print(future_years)


# In[ ]:




