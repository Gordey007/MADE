# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy

# исправить случайное семя для воспроизводимости
seed = 7
numpy.random.seed(seed)

# загрузить набор данных индейцев пима
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# разделить на входные (X) и выходные (Y) переменные
X = dataset[:, 0:8]
Y = dataset[:, 8]

# разделить на 67% для поезда и 33% для теста
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

# создать модель
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Скомпилировать модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Подходит модель
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)