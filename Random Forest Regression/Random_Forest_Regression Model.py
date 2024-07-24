# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Training the Random Forest Regressor on the Training set
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Cross-Validation
cv_scores = cross_val_score(regressor, X, y, cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores)}")

# Plotting Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal (y = x)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual Values')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Values')
plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('House Value')
plt.title('Predicted vs Actual Values for Random Forest Regression')
plt.legend()
plt.show()

# Calculating Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Displaying Model Information
print(f"Model: {regressor}")
print(f"Number of Estimators: {regressor.n_estimators}")
print(f"Max Features: {regressor.max_features}")
print(f"Max Depth: {regressor.max_depth}")
print(f"Min Samples Split: {regressor.min_samples_split}")
print(f"Min Samples Leaf: {regressor.min_samples_leaf}")
print(f"Bootstrap: {regressor.bootstrap}")

# Plotting Feature Importances
importances = regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = dataset.columns[:-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Improving the model by tuning hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

best_regressor = grid_search.best_estimator_

# Retraining the best model on the entire training set
best_regressor.fit(X_train, y_train)

# Predicting the Test set results with the best model
y_pred_best = best_regressor.predict(X_test)

# Plotting Predictions vs Actual Values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal (y = x)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual Values (Best Model)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Values')
plt.scatter(range(len(y_pred_best)), y_pred_best, color='blue', label='Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('House Value')
plt.title('Predicted vs Actual Values for Random Forest Regression (Best Model)')
plt.legend()
plt.show()

# Calculating Metrics for the best model
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Mean Squared Error (Best Model): {mse_best}")
print(f"Mean Absolute Error (Best Model): {mae_best}")
print(f"R^2 Score (Best Model): {r2_best}")
