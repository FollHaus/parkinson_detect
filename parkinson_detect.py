import os  # Импортируем модуль для работы с операционной системой
import pandas as pd  # Импортируем pandas для работы с данными в формате таблиц
import xgboost as xgb  # Импортируем XGBoost для работы с алгоритмом XGBoost
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  # Импортируем RandomForest и StackingClassifier

from sklearn.metrics import accuracy_score  # Импортируем метрику точности для оценки модели
from sklearn.neighbors import KNeighborsClassifier  # Импортируем KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler  # Импортируем MinMaxScaler для нормализации данных
from sklearn.model_selection import GridSearchCV, cross_val_score, \
    RandomizedSearchCV  # Импортируем для поиска гиперпараметров
from sklearn.model_selection import \
    train_test_split  # Импортируем функцию для разделения данных на обучающую и тестовую выборки
from xgboost import XGBClassifier  # Импортируем XGBClassifier для работы с моделью XGBoost

# Устанавливаем количество логических ядер для использования
# Set the number of logical CPUs to use
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

df = pd.read_csv('data/parkinsons.csv')  # Чтение данных из CSV файла
# Load data from CSV file

np_features = df.loc[:, df.columns != 'status'].values[:, 1:]  # Извлекаем все признаки, кроме 'status'
np_target = df.loc[:, 'status'].values  # Целевая переменная - статус

scaler = MinMaxScaler((-1, 1))  # Создаем объект MinMaxScaler для масштабирования признаков
X = scaler.fit_transform(np_features)  # Масштабируем признаки
y = np_target  # Целевые значения

# Разделяем данные на обучающую и тестовую выборки (80%/20%)
# Split data into training and testing sets (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Параметры для поиска (закомментировано для использования позже)
# param_grid = {
#     'n_estimators': [100, 200, 300, 500, 700, 900, 1100],  # Количество деревьев
#     'max_depth': [3, 5, 6, 10, 14, 18],  # Глубина дерева
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Темп обучения
#     'gamma': [0, 0.1, 0.5],  # Параметр для регуляризации
#     'subsample': [0.7, 0.8, 0.9, 1.0],  # Доля выборки для тренировки
#     'colsample_bytree': [0.7, 0.8, 1.0],  # Доля признаков
#     'scale_pos_weight': [1, 2, 3],  # Вес положительных примеров (если есть несбалансированные классы)
# }

# Инициализация модели XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Задаем задачу бинарной классификации
    n_jobs=-1,  # Используем все доступные процессоры
    random_state=42,  # Устанавливаем фиксированное значение для воспроизводимости
    tree_method="hist",  # Метод построения деревьев
    colsample_bytree=0.7,  # Доля признаков для каждого дерева
    gamma=0,  # Параметр для регуляризации
    learning_rate=0.2,  # Темп обучения
    max_depth=3,  # Максимальная глубина дерева
    n_estimators=500,  # Количество деревьев
    scale_pos_weight=1,  # Вес положительных классов
    subsample=0.9,  # Доля обучающей выборки
)

xgb_model.fit(X_train, y_train)  # Обучаем модель XGBoost на обучающей выборке
# Train the XGBoost model on the training data

# # Инициализация GridSearchCV (закомментировано)
# grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
#
# # Обучение модели с использованием GridSearchCV
# grid_search.fit(X_train, y_train)

# Лучшие параметры модели (закомментировано)
# print("Лучшие параметры: ", grid_search.best_params_)

# Использование лучшей модели для предсказания (закомментировано)
# best_model = grid_search.best_estimator_

# Оценка точности модели XGBoost
print(f'Точность на обучающей выборке: {xgb_model.score(X_train, y_train) * 100:.2f}%')
print(f'Точность на тестовой выборке: {xgb_model.score(X_test, y_test) * 100:.2f}%')
# Accuracy on training and testing sets

# Инициализация базовых моделей
knn = KNeighborsClassifier(n_neighbors=5)  # Создаем модель KNN
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)  # Создаем модель XGBoost

# Инициализация мета-модели для стекинга
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Мета-модель - RandomForest

# Создание стекингового классификатора
stacking_clf = StackingClassifier(
    estimators=[('knn', knn), ('xgb', xgb)],  # Базовые модели для стекинга
    final_estimator=meta_model  # Мета-модель для финального предсказания
)

stacking_clf.fit(X_train, y_train)  # Обучение модели стекинга

# Оценка точности на тестовой выборке
y_pred = stacking_clf.predict(X_test)  # Предсказание на тестовых данных
accuracy = accuracy_score(y_test, y_pred)  # Рассчитываем точность
print(f'Точность при использовании нескольких моделей: {accuracy * 100:.2f}%')
# Accuracy when using stacking of multiple models
