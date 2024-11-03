from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
iris = pd.read_csv("/content/gdrive/MyDrive/Colab Notebooks/Iris.csv")
#Encoding the categorical column
iris = iris.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
#Visualize the new dataset
iris.head()
import warnings
warnings.filterwarnings('ignore')
# Check the column names
print(iris.columns)
# Encoding the categorical column
# Using the correct column name "Species"
iris = iris.replace({"Species": {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}})
# Convert the "Species" column to numeric
iris["Species"] = pd.to_numeric(iris["Species"])
# Select only numeric columns
numeric_cols = iris.select_dtypes(include=[np.number]).columns.tolist()
# Compute the correlation matrix
corr_matrix = iris[numeric_cols].corr()
# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation on Iris Species')
plt.show()
X = iris.iloc[:,:-1]
y = iris.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Create the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
#Fit the model for the data
classifier.fit(X_train, y_train)
#Make the prediction
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

