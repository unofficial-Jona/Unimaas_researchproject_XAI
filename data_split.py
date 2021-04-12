import dataset
from sklearn import model_selection


df = dataset.load_data()
train, test = model_selection.train_test_split(df, test_size=0.2)

y_train = train["final_result"].values
X_train = train.drop(["final_result"], axis = 1)
y_test = test["final_result"].values
X_test = test.drop(["final_result"], axis = 1)
