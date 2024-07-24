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
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the KNN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train_class)

# Predicting the Test set results
y_pred_class = classifier.predict(X_test)

# Evaluation Metrics for classification
accuracy = accuracy_score(y_test_class, y_pred_class)
cm = confusion_matrix(y_test_class, y_pred_class)
report = classification_report(y_test_class, y_pred_class)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# Display the summary of the model (optional)
print(f"Model: {classifier}")
print(f"Number of neighbors: {classifier.n_neighbors}")
print(f"Algorithm: {classifier.algorithm}")

# Plotting Confusion Matrix
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting ROC Curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the output
y_test_bin = label_binarize(y_test_class, classes=np.unique(y_class))
y_pred_bin = label_binarize(y_pred_class, classes=np.unique(y_class))
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Plotting Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train_class, cv=5)

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
