import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Define the path to the directory containing the CSV files
path = R"C:\Users\ananya.nakka\Downloads\csv"

# Loop through the CSV files
for i in range(1, 41):
    # read the csv using pandas
    filename = os.path.join(path, f"{i}_raw.csv")
    data = pd.concat([pd.read_csv(filename)])

# separate X and Y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mse', optimizer='adam')

# Define checkpoint to save best model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint])

# Load best model
model = load_model('best_model.h5')

# Evaluate model on validation set
mse = model.evaluate(X_val, y_val)

print("Mean Squared Error on validation set:", mse)

