import numpy as np

from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import model_selection
from sklearn.metrics import f1_score, precision_score, recall_score



input_file = 'income_data.txt'

X = []
y = []

count_class1 = 0
count_class2 = 0


max_datapoints = 5000


with open(input_file, 'r') as f:

    for line in f.readlines():

        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1



# Преобразование в numpy array
X = np.array(X)

print("Количество объектов:", len(X))
print("Класс <=50K:", count_class1)
print("Класс >50K:", count_class2)



# Преобразование строковых данных в числа
label_encoder = []

X_encoded = np.empty(X.shape)


for i, item in enumerate(X[0]):

    if item.isdigit():

        X_encoded[:, i] = X[:, i]

    else:

        label_encoder.append(preprocessing.LabelEncoder())

        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])


X = X_encoded[:, :-1].astype(int)

y = X_encoded[:, -1].astype(int)



# Разделение данных
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=5
)


# Эксперименты
parameters = [

    # Линейное ядро
    ('linear', 1),

    ('linear', 100),

    # RBF ядро
    ('rbf', 1),

    ('rbf', 10)
]


for kernel, C in parameters:

    print()
    print("==============================")
    print("Ядро:", kernel)
    print("C:", C)
    print("==============================")

    # Создание SVM
    if kernel == 'linear':

        classifier = OneVsOneClassifier(
            LinearSVC(
                C=C,
                random_state=0
            )
        )

    else:

        classifier = OneVsOneClassifier(
            SVC(
                kernel='rbf',
                C=C,
                gamma='scale'
            )
        )


    # Обучение
    classifier.fit(X_train, y_train)


    # Предсказание
    y_pred = classifier.predict(X_test)


    # Метрики
    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )


    print("F1 score:", round(100 * f1, 2), "%")

    print("Precision:", round(100 * precision, 2), "%")

    print("Recall:", round(100 * recall, 2), "%")


# Тестовые точки
test_points = [

    [
        '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
        'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
        '0', '0', '40', 'United-States'
    ],

    [
        '28', 'Private', '120000', 'Masters', '14', 'Never-married',
        'Prof-specialty', 'Not-in-family', 'White', 'Female',
        '0', '0', '40', 'United-States'
    ],

    [
        '45', 'Private', '160000', 'Bachelors', '13', 'Married-civ-spouse',
        'Exec-managerial', 'Husband', 'White', 'Male',
        '0', '0', '50', 'United-States'
    ]

]



# Создаём итоговый классификатор
classifier = OneVsOneClassifier(
    LinearSVC(
        C=100,
        random_state=0
    )
)

classifier.fit(X, y)


# Проверяем три тестовые точки
for i, input_data in enumerate(test_points):

    input_data_encoded = [-1] * len(input_data)

    count = 0


    for j, item in enumerate(input_data):

        if item.isdigit():

            input_data_encoded[j] = int(item)

        else:

            input_data_encoded[j] = int(
                label_encoder[count].transform([item])[0]
            )

            count += 1


    input_data_encoded = np.array(input_data_encoded)


    predicted_class = classifier.predict(
        [input_data_encoded]
    )


    result = label_encoder[-1].inverse_transform(
        predicted_class
    )[0]


    print()
    print("Тестовая точка", i + 1)
    print("Результат:", result)