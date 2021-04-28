from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from data_split import X_test, X_train, y_train, y_test



model = MLPClassifier(random_state=1, max_iter=200)
model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)


def compute_cv_score_MLP():
    cv_score = cross_val_score(model, X_train, y_train, cv=10)
    return cv_score

