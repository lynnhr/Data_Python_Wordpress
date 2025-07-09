73% of storage used … If you run out of space, you can't save to Drive or use Gmail. Get 100 GB of storage for $1.99 US$0 for 1 month.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
df = pd.read_csv('vendor-ww-monthly-201006-202406.csv', index_col='Date', parse_dates=True)
df.index.freq = 'MS'

# Decompose time series
results = seasonal_decompose(df['Samsung'])
results.plot()
plt.show()

# Split data
train = df.iloc[:156].copy()
test = df.iloc[156:].copy()

# Scale data
scaler = MinMaxScaler()
scaler.fit(train[['Samsung']])
scaled_train = scaler.transform(train[['Samsung']])
scaled_test = scaler.transform(test[['Samsung']])

# Define generator
n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(generator, epochs=50)

# Make predictions
last_train_batch = scaled_train[-n_input:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))

test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform predictions
true_predictions = scaler.inverse_transform(test_predictions)

# Add predictions to the test DataFrame
test.loc[:, 'Predictions'] = np.nan
test.loc[test.index[-len(true_predictions):], 'Predictions'] = true_predictions

# Plot results
test.plot(figsize=(14, 5))
plt.title('Actual vs. Predicted Values')
plt.xlabel('Date')
plt.ylabel('Samsung')
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(test['Samsung'].dropna(), test['Predictions'].dropna()))
print(f'RMSE: {rmse}')

# Save the test DataFrame with predictions to a CSV file
test.to_csv('Samsungpredictions.csv')
