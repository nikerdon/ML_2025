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
    Gender: bool
    Age: float
    Fare: float
    Region_Code: float
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: float
    Vehicle_Age: int
    Vehicle_Damage: bool
    Driving_License: bool
    Previously_Insured: bool
@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    # Changed here for new variables, may need to be changed again
    new_data = pd.DataFrame({
        'Gender': [input_data.Gender],
        'Age': [input_data.Age],
        "Region_Code": [input_data.Region_Code],
        "Annual_Premium": [input_data.Annual_Premium],
        "Policy_Sales_Channel": [input_data.Policy_Sales_Channel],
        "Vintage": [input_data.Vintage],
        "Vehicle_Age": [input_data.Vehicle_Age],
        "Vehicle_Damage": [input_data.Vehicle_Damage],
        "Driving_License": [input_data.Driving_License],
        "Previously_Insured": [input_data.Previously_Insured]
    })

    # Предсказание
    predictions = model.predict(new_data)

    # Преобразование результата в человеко-читаемый формат
    # Note that we need to decide what to call 1 and 0, as in the file it is just named response
    # I need to check what it is actually predicting
    result = "1" if predictions[0] == 1 else "0"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)