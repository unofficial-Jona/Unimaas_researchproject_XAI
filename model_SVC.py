import main
from sklearn import model_selection
from sklearn.svm import SVC


train, test = model_selection.train_test_split(main.df, test_size=0.2)


y_train = train["final_result"].values
X_train = train.drop(["final_result"], axis = 1)
y_test = test["final_result"].values
X_test = test.drop(["final_result"], axis = 1)

for i in ["linear", "poly", "rbf", "sigmoid"]:  # find best kernel --> linear, poly and rbf are about equal for given data Set
    model = SVC(kernel=i)
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test)
    print(i, score)
