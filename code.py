import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

# Download historical stock price data
df = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

# Visualize the closing price
plt.figure(figsize=(14, 7))
plt.plot(df['Close'])
plt.title('Apple Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price USD')
plt.show()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create a function to prepare the dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Define the time step
time_step = 60

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Create the training and testing datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Calculate Mean Squared Error
mse = np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1)))**2)
print(f"Mean Squared Error: {mse}")

# Plot predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(df.index[train_size + time_step + 1:], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Stock Price')
plt.plot(df.index[train_size + time_step + 1:], predictions, color='red', label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price USD')
plt.legend()
plt.show()
