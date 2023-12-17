import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas_datareader as pdr
from datetime import datetime
import stochasticProcesses

# Parameters for the stochastic processes
# These should be estimated from data but are set to example values here
theta_unemployment = 0.5
mu_unemployment = 5.0
sigma_unemployment = 0.1
mu_cpi = 0.02
sigma_cpi = 0.01
S0_cpi = 100  # Initial CPI value, could be set to the historical starting value
theta_yield = 0.03
mu_yield = 2.0
sigma_yield = 0.05
S0_yield = 0.5  # Initial yield curve value

# Simulation time frame
dt = 1/252
N = 252 * 10  # 10 years of data

# Simulate Unemployment Rate
unemployment_rate = stochasticProcesses.simulate_ou_process(theta_unemployment, mu_unemployment, sigma_unemployment, dt, mu_unemployment, N)

# Simulate CPI
cpi = stochasticProcesses.simulate_gbm_process(mu_cpi, sigma_cpi, S0_cpi, dt, N)

# Simulate Yield Curve
yield_curve = stochasticProcesses.simulate_ou_process(theta_yield, mu_yield, sigma_yield, dt, S0_yield, N)

# Combine the simulations into a feature matrix for the regression model
features = np.column_stack((unemployment_rate, cpi, yield_curve))

# Fetch historical interest rates as the target variable
start = datetime(2010, 1, 1)
end = datetime(2020, 1, 1)
historical_rates = pdr.get_data_fred('FEDFUNDS', start, end).values.flatten()

# Create the regression model and fit it to the historical data
reg_model = LinearRegression()
reg_model.fit(features[:len(historical_rates)], historical_rates)

# Predict interest rates using the fitted model
predicted_interest_rates = reg_model.predict(features)

# Calculate the mean squared error for the model's predictions
mse = mean_squared_error(historical_rates, predicted_interest_rates[:len(historical_rates)])
print(f"Mean Squared Error: {mse}")

# Plot the historical and predicted rates for comparison
plt.figure(figsize=(12, 6))
plt.plot(historical_rates, label='Historical Interest Rates')
plt.plot(predicted_interest_rates[:len(historical_rates)], label='Predicted Interest Rates', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Interest Rate')
plt.title('Interest Rate Prediction vs. Historical Data')
plt.legend()
plt.savefig('prediction.png')  # Save the plot as a PNG file
plt.show()
