# Двоичная классификация с набором данных сонара: базовый уровень
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from keras.initializers import VarianceScaling
from keras.regularizers import l2

from matplotlib import pyplot
import csv

# загрузить набор данных
# dataframe = read_csv("sonar.csv", header=None)
# dataset = dataframe.values

# solution_frame = read_csv("solution-frame.csv", header=None)
# solution_frame = solution_frame.values

datasetTrain = read_csv("train.csv", header=None)
datasetTrain = datasetTrain.values

datasetTrainTarget = read_csv("train-target.csv", header=None)
datasetTrainTarget = datasetTrainTarget.values

datasetTest = read_csv("test.csv", header=None)
datasetTest = datasetTest.values

# разделить на входные (X) и выходные (Y) переменные
# X = dataset[:, 0:60].astype(float)
# Y = dataset[:, 60]

X = datasetTrain[:, 0:30].astype(float)
Y = datasetTrainTarget

# print("X:")
# for x in X:
# 	print(x)
# print("---------------------------------------------------------------------------------------------------------------")
#
# print("Y:")
#
# for y in Y:
# 	print(y)
# print("---------------------------------------------------------------------------------------------------------------")

# кодировать значения класса как целые числа
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)


# базовая модель
def create_smaller():
	# создать модель
	model = Sequential()
	model.add(Dense(13, input_dim=13, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Скомпилировать модель
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# 1
# оценить модель с помощью стандартизированного набора данных
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#
# print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#
# model = create_smaller()

# 2
# определить и подогнать окончательную модель
# model = Sequential()
# model.add(Dense(4, input_dim=13, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(X, Y, epochs=500, verbose=0)

# 3
# определить модель
# model = Sequential()
# model.add(Dense(100, input_dim=13, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # составить модель
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # подходящая модель
# history = model.fit(X, Y, epochs=300, verbose=0)

# 4.1 (4 (13, input_dim=13))
# создать модель
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu', use_bias=True))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Скомпилировать модель
# optimizer='SGD'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, epochs=500, verbose=0)

# # 5
# # создать модель
# model = Sequential()
# model.add(Dense(15, input_dim=30,
# 				kernel_regularizer=l2(0.001),
# 				kernel_initializer=VarianceScaling(),
# 				activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

callbacks = [EarlyStopping(monitor='loss', patience=20), ModelCheckpoint(filepath='best_model.h5', monitor='loss',
																		 save_best_only=True)]

model.fit(X, Y, epochs=500, batch_size=64, verbose=0, callbacks=callbacks)

# новые случаи, когда мы не знаем ответа
Xnew = datasetTest[:, 0:30].astype(float)

# сделать прогноз
yPrediction = model.predict_classes(Xnew)

# показать входы и прогнозируемые результаты
# for i in range(len(Xnew)):
# 	print("X=%s, Prediction=%s" % (i, yPrediction[i]))
with open("Prediction.csv", "w", newline="") as file:
	writer = csv.writer(file)
	writer.writerows(yPrediction)

# сделать прогноз
yPercentPrediction = model.predict_proba(Xnew)

# показать входы и прогнозируемые результаты
# for i in range(len(Xnew)):
# 	print("X=%s, Percent Prediction=%s" % (i, yPercentPrediction[i]))
with open("PercentPrediction.csv", "w", newline="") as file:
	writer = csv.writer(file)
	writer.writerows(yPercentPrediction)

# score = roc_auc_score(test_target, solution_frame)
