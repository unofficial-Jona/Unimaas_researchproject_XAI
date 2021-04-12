from data_split import X_test, X_train, y_train, y_test

from sklearn.ensemble import RandomForestClassifier


def run_random_forest():
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    return accuracy


#Confusion Matrix - Use the whole data X, y without splitting into train, test...
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import confusion_matrix
#Y_pred = cross_val_predict(model, X, y, cv=10)
#print("and Confusion Matrix is")
#print(confusion_matrix(y, Y_pred))