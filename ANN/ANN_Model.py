# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'california_housing.csv')

dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Input

print("ok1")

# Initializing the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Input(shape=(X_train.shape[1],)))
regressor.add(Dense(units=50, activation='relu'))

# Adding the second hidden layer
regressor.add(Dense(units=50, activation='relu'))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the ANN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)

# Predicting the Test set results
y_pred = regressor.predict(X_test).flatten()
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluation Metrics for regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Plotting Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values for ANN')
plt.legend()
plt.show()

# Residual Analysis
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', s=10)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='dashed')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score

def build_regressor():
    regressor = Sequential()
    regressor.add(Input(shape=(X_train.shape[1],)))
    regressor.add(Dense(units=50, activation='relu'))
    regressor.add(Dense(units=50, activation='relu'))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

keras_regressor = KerasRegressor(model=build_regressor, batch_size=10, epochs=100, verbose=0)
cv_scores = cross_val_score(estimator=keras_regressor, X=X, y=y, cv=10, n_jobs=-1)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores)}")

# Model Visualization
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 24)
visualkeras.layered_view(regressor, legend=True, font=font).show()
