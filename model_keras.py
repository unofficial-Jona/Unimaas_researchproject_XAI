from data_split import X_train, X_test, y_test, y_train, df, PrepareDataset
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


data_prep = PrepareDataset(prepare_keras=True)
X_train_ker, X_test_ker, y_train_ker, y_test_ker = data_prep.transform(df)


'''
y_train = np.where(y_train == "Pass", 1, y_train)
y_train = np.where(y_train == "Fail", 0, y_train)
y_train = np.asarray(y_train).astype('float32')

y_test = np.where(y_test == "Pass", 1, y_test)
y_test = np.where(y_test == "Fail", 0, y_test)
y_test = np.asarray(y_test).astype('float32')

'''
model = Sequential()
model.add(Dense(6, input_dim=6, activation="sigmoid"))
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_ker, y_train_ker, epochs=70, use_multiprocessing=True)

results = model.evaluate(X_test_ker,y_test_ker)
print(results)