from data_split import X_test, X_train, y_train, y_test
from sklearn.svm import SVC





def run_SVC():
    best_i = None
    best_score = 0
    for i in ["linear", "poly", "rbf", "sigmoid"]:  # find best kernel --> linear, poly and rbf are about equal for given data Set
        model = SVC(kernel=i)
        model.fit(X_train, y_train)
        score = model.score(X_test,y_test)
        # print(i, score)
        if best_score < score:
            best_score = score
            best_i = i
    return best_i, best_score

# kernel, score = best_kernel()
