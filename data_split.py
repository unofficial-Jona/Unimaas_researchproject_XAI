import dataset
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

def update_df():
    df = dataset.load_data()
    df.to_csv("initialized_dataset", index=False)

# update_df()
df = pd.read_csv('initialized_dataset')


class PrepareDataset(BaseEstimator, TransformerMixin):
    def __init__(self, prepare_nn=False):
        self.prepare_nn = prepare_nn
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
        if self.prepare_nn:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test

data_prep = PrepareDataset()
X_train, X_test, y_train, y_test = data_prep.transform(df)