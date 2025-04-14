import streamlit as st 
import requests
from requests.exceptions import ConnectionError

# ip_api = "127.0.0.1"
ip_api = "model-api"
port_api = "5000"

# Заголовок приложения
st.title("Health Insurance Cross Sell Prediction")

# Ввод данных
st.write("Enter the client details:")

# Client details

selGen = st.selectbox("What is the client's gender:", ['Male', 'Female'])
if selGen == 'Male':
    gender = True
else:
    gender = False

# Текстовое поле для ввода возраста с проверкой на число
ageP = st.text_input("What is the client's age?", value=20)
if not ageP.isdigit():
    st.error("Please enter a valid number for Age.")
    stageA = 0
elif float(ageP) < 16 or float(ageP) > 120:
    # Age range in the data set is 20 to 85
    # 16 years is the driving age in the US. 120 is right above the oldest person,
    # so age can't go higher.
    st.error("Please enter a valid age.")
    stageA = 0
elif float(ageP) < 18 or float(ageP) > 85:
    st.warning("The entered age is outside of the expected parameter range.")
    stageA = 1
else:
    stageA = 1


selLic = st.selectbox("Do they have a driver's license?", ['Yes', 'No'])
if selLic == 'Yes':
    lic = True
else:
    lic = False


rCode = st.text_input("Region Code", value=100)
if not rCode.isdigit():
   st.error("Please enter a valid number for region code.")

# Insurance details

selIns = st.selectbox("Have they been previously insured?", ['Yes', 'No'])
if selIns == 'Yes':
    insured = True
else:
    insured = False

selAgeV = st.selectbox("What is their vehicle's age?", ['< 1 year', '1 - 2 years', '> 2 years'])
if selAgeV == '< 1 year':
    ageV1 = True
    ageV2 = False
elif selAgeV == '1 - 2 years':
    ageV1 = False
    ageV2 = False
else:
    ageV1 = False
    ageV2 = True

selDmg = st.selectbox("Do they have vehicle damage?", ['Yes', 'No'])
if selDmg == 'Yes':
    damage = True
else:
    damage = False

premium = st.text_input("What is their Annual Premium?", value=30000)
if not premium.isdigit():
    st.error("Please enter a valid number for annual premium.")
    stageP = 0
elif float(premium) < 0.0:
    # In dataset: 2630 to 540165
    # Negative numbers don't make sense in context.
    st.error("Please enter a non-negative number for annual premium.")
    stageP = 0
elif float(premium) > 600000:
    # This should be a warning. These values are extrapolated and likely to be 
    # erroneous, but may actually exist.
    st.warning("The entered number is outside of the expected parameter range. Please check your value for annual premium.")
    stageP = 1
else:
    stageP = 1

psc = st.text_input("Policy Sales Channel", value=100)
if not psc.isdigit():
    # Goes from 1 to 163, with gaps.
    st.error("Please enter a whole number between 1 and 180.")
    stageC = 0
elif not float(psc).is_integer():
    # Note that non-integer numbers don't break the model, but seem meaningless
    # if the number is identifying a place or policy.
    st.error("Please enter a whole number between 1 and 180.")
    stageC = 0
elif float(psc) < 1 or float(psc) > 180:
    # 170 chosen as the cutoff based on the idea that perhaps not all 
    # are accounted for in the data, as there are gaps.
    # Going above this number doesn't break the program, but the response
    # will be meaningless if the channel doesn't exist, or the data is 
    # extrapolated too far.
    st.error("Please enter a whole number between 1 and 180.")
    stageC = 0
elif float(psc) > 166:
    st.warning("The entered number is not on the known list of policy sales channels. Check to see if it is correct.")
    stageC = 1
else:
    stageC = 1



vintage = st.text_input("Vintage", value=100)
if not vintage.isdigit():
   st.error("Please enter a valid number for vintage.")



# Кнопка для отправки запроса
if st.button("Predict"):
    # Проверка, что все поля заполнены
    if stageA and stageC and stageP:
        # Подготовка данных для отправки
        data = {
            "Age": float(ageP),
            "Annual_Premium": float(premium),
            "Policy_Sales_Channel": int(float(psc)),
            "Gender_Male": bool(gender),
            "Vehicle_Age_1": bool(ageV1),
            "Vehicle_Age_2": bool(ageV2),
            "Vehicle_Damage_Yes": bool(damage),
            "Driving_License_1": bool(lic),
            "Previously_Insured_1": bool(insured)
        }

        try:
            # Отправка запроса к Flask API
            response = requests.post(f"http://{ip_api}:{port_api}/predict", json=data)

            # Проверка статуса ответа
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Predicted Response: {prediction}")
            else:
                st.error(f"Request failed with status code {response.status_code}")
        except ConnectionError as e:
            st.error(f"Failed to connect to the server")
    else:
        st.error("Please fill in all fields with valid numbers.") 