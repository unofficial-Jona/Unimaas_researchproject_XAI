from data_split import X_train, X_test, y_test, y_train, df, PrepareDataset
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


data_prep = PrepareDataset(prepare_keras=True)
X_train_ker, X_test_ker, y_train_ker, y_test_ker = data_prep.transform(df)


model = Sequential()
model.add(Dense(6, input_dim=6, activation="sigmoid"))
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(1, activation="softmax"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_ker, y_train_ker, epochs=20, use_multiprocessing=True)

results = model.evaluate(X_test_ker,y_test_ker)
