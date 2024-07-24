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

# Convert prices to classes
def convert_to_classes(y, bins):
    y_class = np.digitize(y, bins) - 1
    return y_class

# Define price bins
price_bins = [0, 88700, 119800, 151800, 179800, 217800, 265000, 350700, np.inf]

# Convert y to classes
y_class = convert_to_classes(y, price_bins)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Handle class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_class_res = sm.fit_resample(X_train, y_train_class)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_fscore_support

classifier = GaussianNB()
classifier.fit(X_train_res, y_train_class_res)

# Predicting the Test set results
y_pred_class = classifier.predict(X_test)

# Evaluation Metrics for classification
accuracy = accuracy_score(y_test_class, y_pred_class)
cm = confusion_matrix(y_test_class, y_pred_class)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_class, y_pred_class, average='weighted')
report = classification_report(y_test_class, y_pred_class)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("Classification Report:")
print(report)

# Display the summary of the model
print(f"Model: {classifier}")

# ROC Curve and AUC
y_prob = classifier.predict_proba(X_test)
roc_auc = roc_auc_score(y_test_class, y_prob, multi_class='ovr')

fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(price_bins)-1):
    fpr[i], tpr[i], _ = roc_curve(y_test_class, y_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(price_bins)-1):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(classifier, X_train_res, y_train_class_res, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="g")

plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc="best")
plt.show()

# Cross-Validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(classifier, X, y_class, cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores)}")

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_res, y_train_class_res)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

best_classifier = grid_search.best_estimator_

# Retraining the best model on the entire training set
best_classifier.fit(X_train_res, y_train_class_res)

# Predicting the Test set results with the best model
y_pred_class_best = best_classifier.predict(X_test)

# Evaluation Metrics for the best model
accuracy_best = accuracy_score(y_test_class, y_pred_class_best)
cm_best = confusion_matrix(y_test_class, y_pred_class_best)
precision_best, recall_best, f1_score_best, _ = precision_recall_fscore_support(y_test_class, y_pred_class_best, average='weighted')
report_best = classification_report(y_test_class, y_pred_class_best)

print(f"Accuracy (Best Model): {accuracy_best}")
print("Confusion Matrix (Best Model):")
print(cm_best)
print(f"Precision (Best Model): {precision_best}")
print(f"Recall (Best Model): {recall_best}")
print(f"F1 Score (Best Model): {f1_score_best}")
print("Classification Report (Best Model):")
print(report_best)
