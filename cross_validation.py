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

#Due to the imbalance dataset, we decided to also look at the TPR(True positive rate/Recall) and TNR.
tn, fp, fn, tp = confusion_matrix(y, Y_pred).ravel()
#Model identifies x% of passed students and will miss 1-x% of passed students.
print("TPR for", model, "->", tp / (tp + fn))
#Model identifies x% of failled students and will miss 1-x% of failled students.
print("TNR for", model, "->",  tn / (tn+fp))

#Kappa statistic - reference: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
observed_accuracy = (tn+tp)/(tn+tp+fp+fn)
expected_accuracy = (((tn+fn)*(tn+fp))/(tn+tp+fp+fn)+ ((fp+tp)*(fn+tp))/(tn+tp+fp+fn))/(tn+tp+fp+fn)
kappa_value= (observed_accuracy - expected_accuracy)/(1 - expected_accuracy)
print("Kappa value for", model, "->",  kappa_value)

#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/ for interpretation of kappa value.
#0.76 of kappa value means moderate level of agreement in the data - Data is moderately reliable.