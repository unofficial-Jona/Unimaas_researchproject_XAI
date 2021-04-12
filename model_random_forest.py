from data_split import X_test, X_train, y_train, y_test

from sklearn.ensemble import RandomForestClassifier


def run_random_forest():
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    return accuracy