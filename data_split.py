import dataset
import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin


df = dataset.load_data()
train, test = model_selection.train_test_split(df, test_size=0.2)

y_train_orig = train["final_result"].values
y_train = np.where(y_train_orig == 'Distinction', 'Pass', y_train_orig)
X_train = train.drop(["final_result"], axis = 1)
y_test_orig = test["final_result"].values
y_test = np.where(y_test_orig == 'Distinction', 'Pass', y_test_orig)
X_test = test.drop(["final_result"], axis = 1)


class PrepareDataset(BaseEstimator, TransformerMixin):
    def __init__(self, prepare_keras = False ):
        self.prepare_keras = prepare_keras
    def fit(self, X):
        return self
    def transform(self, df):
        train, test = model_selection.train_test_split(df, test_size=0.2)
        y_train_orig = train["final_result"].values
        y_train = np.where(y_train_orig == 'Distinction', 'Pass', y_train_orig)
        X_train = train.drop(["final_result"], axis=1)
        y_test_orig = test["final_result"].values
        y_test = np.where(y_test_orig == 'Distinction', 'Pass', y_test_orig)
        X_test = test.drop(["final_result"], axis=1)
        if self.prepare_keras:
            y_train = np.where(y_train == 'Pass', 1, y_train)
            y_train = np.where(y_train == 'Fail', 0, y_train)
            y_train = np.asarray(y_train).astype('float32')
            y_test = np.where(y_test == "Pass", 1, y_test)
            y_test = np.where(y_test == "Fail", 0, y_test)
            y_test = np.asarray(y_test).astype('float32')

            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test
