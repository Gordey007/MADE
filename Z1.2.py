# многослойная модель персептрона для задачи двух окружностей
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# # генерировать набор данных
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# разделить на поезд и проверить
n_test = 500
trainX, testX = X[:n_test, :], X[n_test:, :]
trainy, testy = y[:n_test], y[n_test:]

# print("trainX")
# for n in trainX:
#     print(n)
#
# print("trainy")
# for n in trainy:
#     print(n)
#
# print("testX")
# for n in testX:
#     print(n)
#
# print("testy")
# for n in testy:
#     print(n)

# определить модель
model = Sequential()
model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# составить модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# подходящая модель
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)

# оценить модель
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# потеря сюжета во время тренировки
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# точность сюжета во время тренировки
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
