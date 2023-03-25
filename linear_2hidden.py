import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Create an empty list to store the MSE values
mse_list = []

# Define the path to the directory containing the CSV files
path = R"C:\Users\satish.nakka\Downloads\csv"

# Loop through the CSV files
for i in range(1, 41):
    # read the csv using pandas
    filename = os.path.join(path, f"{i}_raw.csv")
    #data = pd.read_csv(R"C:\Users\satish.nakka\Downloads\csv\filename")
    data = pd.read_csv(filename)

    # separate X and Y
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # split the dataset into train and test
    X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=42)

    # Split train set into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    # Print the feature matrix order for the training set
    #print(f'The feature matrix order for the training set is: {X_train.shape}')

    # Print the feature matrix order for the test set
    #print(f'The feature matrix order for the test set is: {X_test.shape}')

    # train the model
    #model = LinearRegression()
    # Print the shapes of the data subsets
    print("Training set - X shape:", X_train.shape, " Y shape:", Y_train.shape)
    print("Validation set - X shape:", X_val.shape, " Y shape:", Y_val.shape)
    print("Test set - X shape:", X_test.shape, " Y shape:", Y_test.shape)

    # Print the first few rows of each subset to verify the split
    print("Training set:")
    print(X_train.head())
    print(Y_train.head())

    print("Validation set:")
    print(X_val.head())
    print(Y_val.head())

    print("Test set:")
    print(X_test.head())
    print(Y_test.head())

    # Train an MLPRegressor model with two hidden layers each with 100 neurons
    model = MLPRegressor(hidden_layer_sizes=(100, 100))

    #Train an MLPRegressor model with one hidden layers with 100 neurons
    #model = MLPRegressor(hidden_layer_sizes=(100,))

    model.fit(X_train, Y_train)

    # predict on the test set
    Y_pred = model.predict(X_test)

    print("Predict set:")
    print(X_test.head())
    print("Y_pred:", Y_pred)
    print(Y_test.head())

    # measure the performance
    mse = mean_squared_error(Y_test, Y_pred)

    # Add the MSE value to the list
    mse_list.append(mse)

    print('MSE:', i, mse)

# Calculate the mean of the MSE values
mean_mse = sum(mse_list) / len(mse_list)
