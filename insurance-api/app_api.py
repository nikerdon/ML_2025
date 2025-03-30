'''
Давайте создадим простое API с тремя ручками: одна для предсказания выживания (/predict), 
другая для получения количества сделанных запросов (/stats), и третья для проверки работы API (/health).

Шаг 1: Установка необходимых библиотек
Убедитесь, что у вас установлены необходимые библиотеки:
pip install fastapi uvicorn pydantic scikit-learn pandas

Шаг 2: Создание app_api.py
Шаг 3: Запустите ваше приложение: python app_api.py
Шаг 4: Тестирование API
Теперь вы можете протестировать ваше API с помощью curl или любого другого инструмента для отправки HTTP-запросов.

Проверка работы API (/health)
curl -X GET http://127.0.0.1:5000/health
curl -X GET http://127.0.0.1:5000/stats
curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Pclass\": 3, \"Age\": 22.0, \"Fare\": 7.2500}"
'''

from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели из файла pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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

def transform_data(dataset):
    ds_new = dataset.to_numpy()

    # transform to standard scalar (note that we need to load this data)
    # mean: 38.38755612, 30468.54795689, 112.40399492
    # scale: 1.49967598e+01, 1.64761977e+04, 5.40481829e+01
    age_mean, ap_mean, psc_mean = 38.38755612, 30468.54795689, 112.40399492
    age_std, ap_std, psc_std = 14.9967598, 1.64761977e04, 54.0481829
    ds_new[1] = (ds_new[1] - age_mean) / age_std
    ds_new[2] = (ds_new[2] - ap_mean) / ap_std
    ds_new[3] = (ds_new[3] - psc_mean) / psc_std

    # transform to principle components

    return ds_new

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    # Changed here for new variables, may need to be changed again
    new_data = pd.DataFrame({
        'Gender_Male': [input_data.Gender_Male],
        'Age': [input_data.Age],
        #"Region_Code": [input_data.Region_Code],
        "Annual_Premium": [input_data.Annual_Premium],
        "Policy_Sales_Channel": [input_data.Policy_Sales_Channel],
        #"Vintage": [input_data.Vintage],
        "Vehicle_Age_< 1 Year": [input_data.Vehicle_Age_1],
        "Vehicle_Age_> 2 Years": [input_data.Vehicle_Age_2],
        "Vehicle_Damage_Yes": [input_data.Vehicle_Damage_Yes],
        "Driving_License_1": [input_data.Driving_License_1],
        "Previously_Insured_1": [input_data.Previously_Insured_1]
    })

    # this assumes that the model works with the normalized data
    # and has not information of the original data
    # I believe this is the case
    ds_new = transform_data(new_data)

    # Предсказание
    predictions = model.predict(ds_new)

    # Преобразование результата в человеко-читаемый формат
    # Note that we need to decide what to call 1 and 0, as in the file it is just named response
    # I need to check what it is actually predicting
    result = "Positive" if predictions[0] == 1 else "Negative"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)