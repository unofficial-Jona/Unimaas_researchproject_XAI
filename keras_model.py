from data_split import X_train, X_test, y_test, y_train, df
from keras.models import Sequential
from keras.layers import Dense
import numpy as np



y = df["final_result"]
y = np.where(y == "Distinction", "Pass", y)
y = y[1:]
y = np.where(y == "Pass", 1, y)
y = np.where(y == "Fail", 0, y)
y = np.asarray(y).astype('float32')
X = df.drop("final_result", axis=1)
X = np.array(X)
X = np.delete(X,0,0)



model = Sequential()
model.add(Dense(6, input_dim=6, activation="sigmoid"))
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, use_multiprocessing=True)
print(model.summary())