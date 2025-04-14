'''
Давайте создадим простое API с 4 ручками: одна для предсказания выживания (/predict), 
другая для получения количества сделанных запросов (/stats), третья для проверки работы API (/health) и четвертая - тест модели в api .

Шаг 1: Установка необходимых библиотек
Убедитесь, что у вас установлены необходимые библиотеки:
pip install fastapi uvicorn pydantic scikit-learn pandas

Note: may need upgrade and force reinstall
pip install --upgrade --force-reinstall <package>
pip install -I <package>
pip install --ignore-installed <package>

Шаг 2: Создание app_api.py
Шаг 3: Запустите ваше приложение: python app_api.py
Шаг 4: Тестирование API
Теперь вы можете протестировать ваше API с помощью curl или любого другого инструмента для отправки HTTP-запросов.

Проверка работы API (/health)
curl -X GET http://127.0.0.1:5000/health
curl -X GET http://127.0.0.1:5000/stats
curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Pclass\": 3, \"Age\": 22.0, \"Fare\": 7.2500}"
'''

# import libraries
from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели из файла pickle
# with open('./Итог/api/Gradient_Boosting_Classifier_Model.pkl', 'rb') as f:
with open('bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# transformer of data
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# for rescaling data
# three posibilities: standard_scale object, csv file with data of mean, then std, or just adding the data here
# the third variant is not recommended

with open('scaler.pkl', 'rb') as f:
   scaler = pickle.load(f)

# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
# Changed here for new variables, but may need to be changed again
class PredictionInput(BaseModel):
    Gender_Male: bool
    Age: float
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vehicle_Age_1: bool
    Vehicle_Age_2: bool
    Vehicle_Damage_Yes: bool
    Driving_License_1: bool
    Previously_Insured_1: bool

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/test")
def test():
    return {"status": "OK"}

def transform_data(dataset):
    numeric_features = ["Age", 'Annual_Premium', 'Policy_Sales_Channel']
    dataset[numeric_features] = scaler.transform(dataset[numeric_features])

    ds_new = pca.transform(dataset)

    return ds_new

# Structure copied from example code. Receive parameters and predict responses. 
# Note that the structure here could enable us to input many instances to predict at
# once, but I don't want to put that feature in 
@app.post("/predict")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'Age': [input_data.Age],
        "Annual_Premium": [input_data.Annual_Premium],
        "Policy_Sales_Channel": [input_data.Policy_Sales_Channel],
        'Gender_Male': [input_data.Gender_Male],
        "Vehicle_Age_< 1 Year": [input_data.Vehicle_Age_1],
        "Vehicle_Age_> 2 Years": [input_data.Vehicle_Age_2],
        "Vehicle_Damage_Yes": [input_data.Vehicle_Damage_Yes],
        "Driving_License_1": [input_data.Driving_License_1],
        "Previously_Insured_1": [input_data.Previously_Insured_1]
    })

    # test data needs to be transformed to match the form of the training data
    ds_new = transform_data(new_data)

    # Предсказание
    predictions = model.predict(ds_new)
    # predictions = model.predict(ds_new[:, 0:6])

    
    # Интропретация результата
    result = "Positive" if predictions[0] == 1 else "Negative"

    return {"prediction": result}

# host = 0.0.0.0 for docker, 127.0.0.1 without docker
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    # uvicorn.run(app, host="127.0.0.1", port=5000)