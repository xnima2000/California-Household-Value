# California Housing Price Prediction

This repository contains various machine learning models implemented to estimate housing prices in California. The models included are KNN, ANN, Linear Regression, Polynomial Regression, Random Forest Regression, and Naive Bayes for now. Each model is placed in its own folder with the corresponding dataset and performance images.

## Repository Structure

- `ANN/`
  - `ANN_Model.py`: Code for Artificial Neural Network model.
  - `california_housing.csv`: Dataset used for the ANN model.
  - `california_housing_train.csv`: Dataset used for training the ANN model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the ANN model (splitted manually).
  - `Performance - Predicted vs Actual Values.png`: Predicted vs Actual Values image of the ANN model.
  - `Performance - Residual Plot.png`: Residual Plot image of the ANN model.
  - `Performance - Report.txt`: Descriptive other performance evaluation for the ANN model.

- `KNN - Classification/`
  - `KNN_Classification_Model.py`: Code for K-Nearest Neighbors Classification model.
  - `california_housing.csv`: Dataset used for the KNN Classification model.
  - `california_housing_train.csv`: Dataset used for training the KNN Classification model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the KNN Classification model (splitted manually).
  - `KNN_Classification_Confusion_Matrix.png`: Performance Confusion Matrix of the KNN Classification model.
  - `KNN_Classification_Learning_Curve_CV_and_train.png`: Performance Learning Curve Cross-Validation and Training of the KNN Classification
  - `KNN_Classification_Receiver_Operating_Characteristic_(ROC)_Curves.png`: Performance ROC Curve of the KNN Classification
  - `KNN_Classification_report.txt`: Descriptive other performance evaluation for the KNN model.

- `KNN - Regression/`
  - `KNN_Regression_Model.py`: Code for K-Nearest Neighbors Regression model.
  - `california_housing.csv`: Dataset used for the KNN Regression model.
  - `california_housing_train.csv`: Dataset used for training the KNN Regression model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the KNN Regression model (splitted manually).
  - `KNN_Regression_Residual_Analysis.png`: Performance Residual of the Regression KNN model.
  - `KNN_Regression_Learning_Curve.png`: Performance Learning Curve Cross-Validation and Training of the KNN Regression
  - `KNN_Regression_Plotting_Predictions_vs_Actual_Values`: Ploting Predict vs Actual of the KNN Regression
  - `KNN_Regression_report.txt`: Descriptive other performance evaluation for the KNN Regression model.

- `Linear_Regression/`
  - `Linear_Regression_model.py`: Code for Linear Regression model.
  - `california_housing.csv`: Dataset used for the Linear Regression model.
  - `california_housing_train.csv`: Dataset used for training the Linear Regression model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the Linear Regression model (splitted manually).
  - `Linear_Regression_Learning_Curve.png`: Performance Learning Curve of the Linear Regression model.
  - `Linear_Regression_Predicted_vs_Actual_Values.png`: Performance Predicted vs Actual Values of the Linear Regression model.
  - `Linear_Regression_Residual_Plot.png`: Performance Residual of the Linear Regression model.
  - `Linear_Regression_report.txt`: Descriptive other performance evaluation for the Linear Regression model.

- `Naive_Bayes/`
  - `Naive_Bayes_model.py`: Code for Naive Bayes model.
  - `california_housing.csv`: Dataset used for the Naive Bayes model.
  - `california_housing_train.csv`: Dataset used for training the Naive Bayes model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the Naive Bayes model (splitted manually).
  - `Naive_Bayes_Learning_Curve.png`: Performance Learning Curve of the Naive Bayes.
  - `Naive_Bayes_Receiver_Operating_Characteristic_(ROC)_Curves.png`: Performance ROC Curve of the Naive Bayes
  - `Naive_Bayes report.txt`: Descriptive other performance evaluation for the Naive Bayes model.

- `Polynomial_Regression/`
  - `Polynomial_Regression_model.py`: Code for Polynomial Regression model.
  - `california_housing.csv`: Dataset used for the Polynomial Regression model.
  - `california_housing_train.csv`: Dataset used for training the Polynomial Regression model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the Polynomial Regression model (splitted manually).
  - `Polynomial_Regression_Predicted_vs_Actual_Values.png`: Performance Predicted vs Actual Values of the Polynomial Regression model.
  - `Polynomial_Regression_Learning_Curve.png`: Performance Learning Curve of the Polynomial Regression.
  - `Polynomial_Regression_Residual_Plot.png`: Performance Residual of the Polynomial Regression model.
  - `Polynomial_Regression report.txt`: Descriptive other performance evaluation for the Polynomial Regression model.

- `Random_Forest_Regression/`
  - `Random_Forest_Regression_model.py`: Code for Random Forest Regression model.
  - `california_housing.csv`: Dataset used for the Random Forest Regression model.
  - `california_housing_train.csv`: Dataset used for training the Random Forest Regression model (splitted manually).
  - `california_housing_test.csv`: Dataset used for testing the Random Forest Regression model (splitted manually).
  - `Random_Forest Feature Importance.png`: Performance Feature Importance of the Random Forest Regression model.
  - `Random_Forest Predicted vs Actual Values (Best Model) (AFTER GS).png`: Performance Predicted vs Actual Values (Best Model) of the Random Forest Regression model After GS.
  - `Random_Forest Predicted vs Actual Values sample (Best Model) (AFTER GS).png`: Performance  of the Random Forest Regression model.
  - `Random_Forest Predicted vs Actual Values Sample BEFORE GS.png`: Performance  of the Random Forest Regression model.
  - `Random_Forest_Predictions vs Actual Values BEFORE GS.png`: Performance  of the Random Forest Regression model.
  - `Random_Forest_Regression report.txt`: Descriptive other performance evaluation for the Random Forest Regression model.


## Models Description

### K-Nearest Neighbors (KNN)
- **Description:** This model uses the KNN algorithm for classification and regression tasks. It predicts the price category of the houses based on the nearest neighbors.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

### Artificial Neural Network (ANN)
- **Description:** This model employs a neural network with multiple hidden layers to predict housing prices.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

### Linear Regression
- **Description:** A simple linear regression model to predict housing prices.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

### Polynomial Regression
- **Description:** Extends linear regression by adding polynomial features to capture non-linear relationships.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

### Random Forest Regression
- **Description:** This model uses an ensemble of decision trees (random forest) to predict housing prices.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

### Naive Bayes
- **Description:** Applies the Naive Bayes algorithm for regression tasks.
- **Performance:** [Link to performance image]
- **Dataset:** `california_housing.csv`

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xnima2000/California-Household-Value.git
   ```

2. **Navigate to the specific model folder:**
   ```bash
   cd California-Housing-Price-Estimation/KNN
   ```

3. **Install the necessary dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the model:**
   ```bash
   python KNN_model.py
   ```

## Dependencies

Ensure you have the following libraries installed:
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow (for ANN)

You can install them using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

## Results

Each model folder contains an image showing the performance of the model, comparing predicted vs actual values, and various evaluation metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.

## Contributing

Feel free to contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact nima2abdi@gmail.com.
