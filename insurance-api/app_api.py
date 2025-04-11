'''
Backend for prediction from model. Code modified from example in class. 
Instructions unchanged.

Last edit: 06.04.2025 -- Elizabeth Gould

Давайте создадим простое API с тремя ручками: одна для предсказания выживания (/predict), 
другая для получения количества сделанных запросов (/stats), и третья для проверки работы API (/health).

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
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# transformer of data
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# for rescaling data
# three posibilities: standard_scale object, csv file with data of mean, then std, or just adding the data here
# the third variant is not recommended

with open('scaler.pkl', 'rb') as f:
   scaler = pickle.load(f)

#msd = pd.read_csv('mean_std.csv')
#age_mean, ap_mean, psc_mean = msd.at(0, 'Age'), msd.at(0, 'Annual_Premium'), msd.at(0, 'Policy_Sales_Channel')
#age_std, sp_std, psc_std = msd.at(1, 'Age'), msd.at(1, 'Annual_Premium'), msd.at(1, 'Policy_Sales_Channel')

# mean: 38.38755612, 30468.54795689, 112.40399492
# scale: 1.49967598e+01, 1.64761977e+04, 5.40481829e+01
# age_mean, ap_mean, psc_mean = 38.38755612, 30468.54795689, 112.40399492
# age_std, ap_std, psc_std = 14.9967598, 1.64761977e04, 54.0481829

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

# In order to use the model to predict, we need to transform the data by the identical transformation
# as was used to train the data. Make sure that you only transform, and not refit, as that will change
# the transformation and therefore not provide the correct transformation.
def transform_data(dataset):
    #ds_new = dataset.to_numpy()

    # transform to standard scalar from saved means and standard deviations
    #dataset['Age'] = (dataset['Age'] - age_mean) / age_std
    #dataset['Annual_Premium'] = (dataset['Annual_Premium'] - ap_mean) / ap_std
    #dataset['Policy_Sales_Channel'] = (dataset['Policy_Sales_Channel'] - psc_mean) / psc_std

    # transform to standard scalar with pickled scaler
    # This is the more faithful variant.
    numeric_features = ["Age", 'Annual_Premium', 'Policy_Sales_Channel']
    dataset[numeric_features] = scaler.transform(dataset[numeric_features])

    # use pca analysis
    # Remove this code if the model requires the original data.
    ds_new = pca.transform(dataset)

    return ds_new
    return dataset

# Structure copied from example code. Receive parameters and predict responses. 
# Note that the structure here could enable us to input many instances to predict at
# once, but I don't want to put that feature in 
@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    # Changed here for new variables, may need to be changed again
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
    #predictions = model.predict(ds_new)
    predictions = model.predict(ds_new[:, 0:6])

    
    # Преобразование результата в человеко-читаемый формат
    # Note that we need to decide what to call 1 and 0, as in the file it is just named response
    # I need to check what it is actually predicting
    result = "Positive" if predictions[0] == 1 else "Negative"
    #strng = f'gen: {ds_new["Gender_Male"][0]}, age: {ds_new["Age"][0]}, ap: {ds_new["Annual_Premium"][0]}, psc: {ds_new["Policy_Sales_Channel"][0]}, va1: {ds_new["Vehicle_Age_< 1 Year"][0]}, va2: {ds_new["Vehicle_Age_> 2 Years"][0]}, vd: {ds_new["Vehicle_Damage_Yes"][0]}, dl: {ds_new["Driving_License_1"][0]}, pi {ds_new["Previously_Insured_1"][0]}' 
    #strng = f'1: {ds_new[0,0]}, 2: {ds_new[0,1]}, 3: {ds_new[0,2]}, 4: {ds_new[0,3]}, 5: {ds_new[0,4]}, 6: {ds_new[0,5]}' 

    return {"prediction": result}
    #return {"prediction": strng}

# host = 0.0.0.0 for docker, 127.0.0.1 without docker
if __name__ == '__main__':
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=5000)
    uvicorn.run(app, host="127.0.0.1", port=5000)