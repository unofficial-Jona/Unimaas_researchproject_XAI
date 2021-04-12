import dataset
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#model = KNeighborsClassifier(n_neighbors=5)
model = SVC(kernel='poly')
df = dataset.load_data()
y = df["final_result"].values
y = np.where(y == 'Distinction', 'Pass', y)
X = df.drop(["final_result"], axis = 1)

#Model Validation via K-fold Cross-Validation
results_kfold = model_selection.cross_val_score(model, X, y, cv=10)
print("Accuracy for", model, "->", (results_kfold.mean() * 100.0))

#Confusion Matrix
Y_pred = cross_val_predict(model, X, y, cv=10)
print("and Confusion Matrix is")
print(confusion_matrix(y, Y_pred))