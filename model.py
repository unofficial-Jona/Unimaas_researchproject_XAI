import main
from sklearn import model_selection
from sklearn.svm import SVC


train, test = model_selection.train_test_split(main.df, test_size=0.2)

# X_train = train["num_of_prev_attempts", "weighted_grade", "pass_rate", "exam_score", "date", "sum_click"].values

y_train = train["final_result"].values
X_train = train.drop(["final_result"], axis = 1)
y_test = test["final_result"].values
X_test = test.drop(["final_result"], axis = 1)


model = SVC(kernel="linear")
model.fit(X_train, y_train)