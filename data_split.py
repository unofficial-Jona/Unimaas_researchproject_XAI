import dataset
import numpy as np
from sklearn import model_selection


df = dataset.load_data()
train, test = model_selection.train_test_split(df, test_size=0.2)

y_train_orig = train["final_result"].values
y_train = np.where(y_train_orig == 'Distinction', 'Pass', y_train_orig)
X_train = train.drop(["final_result"], axis = 1)
y_test_orig = test["final_result"].values
y_test = np.where(y_test_orig == 'Distinction', 'Pass', y_test_orig)
X_test = test.drop(["final_result"], axis = 1)
