import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time

#Функция для перевода строки новых ненормализованных данных в строку нормализованных
# features здесь это только числовые параметры, в том же порядке в котором они записаны в scaler (а значит в массив "numeric_features")
def new_scaler(features, scale, mean):
    for i in range(len(features)):
        features[i] = (features[i] - mean[i])/ scale[i]
    return features

#Функция для перевода строки нормализованных данных в первоночальный вид
def scaled_to_normal(features, scale, mean):
    for i in range(len(features)):
        features[i] = features[i] * scale[i] + mean[i]
    return features

#Чтение данных

data = pd.read_csv('train_short.csv')
# Показать названия всех столбцов
print("\nНазвания столбцов:")
print(data.columns.tolist())

# Показать первые несколько строк
print("\nПервые 2 строк данных:")
print(data.head(2))

#data = data.head(1000)

# Удаляем строки с пропущенными значениями (их нет)
data = data.dropna()

dataT = pd.read_csv('test_short.csv')
dataT = dataT.dropna()


#Оставляем только нужные столбцы
#columns_to_keep = ['Response', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
columns_to_keep = ['Response', 'Gender', 'Age', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel']
data = data[columns_to_keep]
dataT = dataT[columns_to_keep]

# One-hot кодирование категориальных признаков
categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Driving_License', 'Previously_Insured']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
dataT = pd.get_dummies(dataT, columns=categorical_features, drop_first=True)


length = len(data)

# Нормализация числовых признаков
#numeric_features = ["Age", 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
numeric_features = ["Age", 'Annual_Premium', 'Policy_Sales_Channel']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])
#print(data.head(2))

dataT[numeric_features] = scaler.transform(dataT[numeric_features])

#Функция для удаления выбросов. Проверяем только численные значения
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    #Можно тут поиграться с коэффициентом на который умножается
    lower_bound = Q1 - 1.6 * IQR
    upper_bound = Q3 + 1.6 * IQR
    
    return df[(df[column] >= lower_bound) & 
             (df[column] <= upper_bound)]

data_clean = data
#Удаление выбросов
for col in data_clean.select_dtypes(include=['float64']).columns:
    data_clean = remove_outliers_iqr(data_clean, col)

print("Строчек удалено: ", len(data) - len(data_clean))
print(len(data_clean))
print(len(data))

data = data_clean

print(data.Response.value_counts())

# Разделение данных
df_majority = data[data['Response'] == 0]  # Мажоритарный класс (0)
df_minority = data[data['Response'] == 1]  # Миноритарный класс (1)

# Определяем, сколько строк оставить в классе 0 (в 2 раза больше, чем класс 1)
n_samples = 1 * len(df_minority)  # 2:1 соотношение

# Случайно выбираем подмножество
df_majority_downsampled = resample(
    df_majority,
    replace=False,      # Без повторяющихся строк
    n_samples=n_samples,
    random_state=42     # Для воспроизводимости
)

# Объединяем с миноритарным классом
data = pd.concat([df_majority_downsampled, df_minority])

# Проверяем соотношение
print(data['Response'].value_counts())



# Разделение на признаки (X) и целевую переменную (y)
X_train = data.drop('Response', axis=1)
y_train = data['Response']

X_test = dataT.drop('Response', axis=1)
y_test = dataT['Response']


# Разделение на обучающую и тестовую выборки. Только для проверки PCA, потом надо заменить на кросс-валидацию
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

n = 6
# Создаем объект PCA с n компонентами
pca = PCA(n_components=n)

# Применяем PCA к данным
pca_data = pca.fit_transform(X_train)

# Создаем DataFrame с результатами
pca_df = pd.DataFrame(
    pca_data,
    columns=[f'Фактор_{i+1}' for i in range(n)]
)



X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Naive Bayes

# parameters to provide
model_filename  = f"bayes_model.pkl"

# run model, with timing
t_start = time.time()
test_model = GaussianNB()
test_model.fit(X_train, y_train)
y_pred = test_model.predict(X_test)
t_finish = time.time()
dt = t_finish - t_start
dt2 = dt / 60.
print("Time: ",  dt, " sec or ", dt2, "min")
print(classification_report(y_test, y_pred))
# Сохранение модели в файл
with open(model_filename, 'wb') as file:
    pickle.dump(test_model, file)
print(f"Модель сохранена в файл: {model_filename}")

with open("pca.pkl", 'wb') as file:
    pickle.dump(pca, file)
print(f"Модель сохранена в файл: {model_filename}")

with open("scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)
print(f"Модель сохранена в файл: {model_filename}")

