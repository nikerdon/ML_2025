import streamlit as st 
import requests
from requests.exceptions import ConnectionError

ip_api = "insurance-api"
port_api = "5000"

# Заголовок приложения
st.title("Insurance Cross Sales Prediction")

'''
# for clearing boxes when license is 0

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage

#variables here = initial values for all numericals, lic = false
'''

# Ввод данных
st.write("Enter the client details:")


# I need to match the variables here to how they are read by the model.

# Client details

# I have this first, because I am pretty sure I can clear the other
# fields if the answer is yes, and just return 'no' for the response
selLic = st.selectbox("Do they have a driver's license?", ['Yes', 'No'])
if selLic == 'Yes':
    lic = True
else:
    lic = False

#submit_button = st.form_submit_button(label='Submit', on_click=set_stage, args=(1,))
#I could put this into the if statement above, clearing everything when license is changed

'''
# more box clear code examples / templates
if st.session_state.stage > 0:
    var = st.input(params)
    st.button('Submit', on_click=set_stage, args=(2,))

if st.session_state.stage > 1:
    st.write(result)
'''

# If we stick to English, we can use either the word gender (род) or sex (пол)
# My guess is gender is better here, since this is not medicine, sport, or official paperwork
selGen = st.selectbox("What is the client's gender:", ['Male', 'Female'])
if selGen == 'Male':
    gender = True
else:
    gender = False

# Текстовое поле для ввода возраста с проверкой на число
ageP = st.text_input("What is the client's age?", value=10)
if not ageP.isdigit():
    st.error("Please enter a valid number for Age.")

# Insurance details

selIns = st.selectbox("Have they been previously ensured?", ['Yes', 'No'])
if selLic == 'Yes':
    insured = True
else:
    insured = False

premium = st.text_input("Annual Premium", value=100)
if not premium.isdigit():
    st.error("Please enter a valid number for annual premium.")

# Car details

selAgeV = st.selectbox("What is their vehicle's age?", ['< 1 year', '1 - 2 years', '> 2 years'])
if selAgeV == '< 1 year':
    ageV = 0
elif selAgeV == '1 - 2 years':
    ageV = 1
else:
    ageV = 2

vintage = st.text_input("Vintage", value=100)
if not vintage.isdigit():
    st.error("Please enter a valid number for vintage.")

selDmg = st.selectbox("Do they have vehicle damage?", ['Yes', 'No'])
if selDmg == 'Yes':
    damage = True
else:
    damage = False


# Location Codes

rCode = st.text_input("Region Code", value=100)
if not rCode.isdigit():
    st.error("Please enter a valid number for region code.")
    
psc = st.text_input("Policy Sales Channel", value=100)
if not psc.isdigit():
    st.error("Please enter a valid number for policy sales channel.")

    


# Кнопка для отправки запроса
if st.button("Predict"):
    # This is the variant which is immediate
    # Can be used or removed
    if lic == False:
        prediction = 0
        st.success(f"Prediction: {prediction}")
        # Проверка, что все поля заполнены
    elif ageP.isdigit() and premium.isdigit() and vintage.isDigit() and rCode.isdigit() and psc.isdigit():
        # Подготовка данных для отправки
        data = {
            "Gender": bool(gender),
            "Age": float(ageP),
            "Region_Code": float(rCode),
            "Annual_Premium": float(premium),
            "Policy_Sales_Channel": float(psc),
            "Vintage": float(vintage),
            "Vehicle_Age": int(ageV),
            "Vehicle_Damage": bool(damage),
            "Driving_License": bool(license),
            "Previously_Insured": bool(insured)
        }

        try:
            # Отправка запроса к Flask API
            response = requests.post(f"http://{ip_api}:{port_api}/predict_model", json=data)

            # Проверка статуса ответа
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Request failed with status code {response.status_code}")
        except ConnectionError as e:
            st.error(f"Failed to connect to the server")
    else:
        st.error("Please fill in all fields with valid numbers.")