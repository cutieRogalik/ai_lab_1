import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Перемешивание данных
X, y = shuffle(data, target, random_state=7)

# Разбивка данных на обучающий и тестовый наборы
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Создание регрессионной модели на основе SVМ
sv_regressor = SVR(kernel='linear', C=100.0, epsilon=0.3)

# Обучение регрессора SVМ
sv_regressor.fit(X_train, y_train)

# Оценка эффективности работы регрессора
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Тестирование регрессора на тестовой точке данных
test_data = [1.5, 0, 12.5, 0, 0.55, 6.8,
    70, 4.5, 5, 350, 16.5, 380, 8.0]
print("\nPredicted price:", sv_regressor.predict([test_data])[0])