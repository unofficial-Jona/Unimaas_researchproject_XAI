from data_split import X_test, X_train, y_train, y_test
from sklearn.svm import SVC


def run_SVC():
    best_i = None
    best_score = 0
    for i in ["linear", "poly", "rbf", "sigmoid"]:  # find best kernel --> linear, poly and rbf are about equaSet
        model = SVC(kernel=i)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # print(i, score)
        if best_score < score:
            best_score = score
            best_i = i
    return best_i, best_score

# kernel, score = best_kernel()

#Confusion Matrix - Use the whole data X, y without splitting into train, test
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import confusion_matrix
#Y_pred = cross_val_predict(model, X, y, cv=10)
#print("and Confusion Matrix is")
#print(confusion_matrix(y, Y_pred))