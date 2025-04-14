# ML_2025
# Binary Classification of Insurance Cross Selling
---

This project aims to predict customer responses to automobile insurance offers using machine learning. An API was developed to serve predictions, and the solution was containerized with Docker for seamless deployment.
The dataset used: https://www.kaggle.com/competitions/playground-series-s4e7/overview


# How to Use
---

An API for local work needs to be built using the command docker-compose -p byaess_model up


# Preprocessing
---

A check for empty values ​​was performed; outliers were removed using the IQR method. A correlation matrix was constructed to select the parameters used. Normalization and centering of numerical data and one-hot coding of categorical features were performed. Principal сomponents analysis was performed.

The data were balanced by removing an excessive number of rows with a negative answer. Models built on such data perform better than models built on the entire sample.


# Training and Testing
---

Results:

| Model            | F1   | recall | precision |
|------------------|------|--------|-----------|
| Bagging          | 0.69 | 0.67   | 0.72      |
| GradientBoosting | 0.82 | 0.92   | 0.74      |
| AdaBoost         | 0.82 | 0.92   | 0.73      |
| Naive Bayes      | 0.81 | 0.90   | 0.87      |


# Backend and Frontend
---
Backend: python app_api.py
Frontend: streamlit run streamlit_app.py

Step 1: Installing Required Libraries
Ensure you have the necessary libraries installed:  

> pip install fastapi uvicorn pydantic scikit-learn pandas

Note: You may need to upgrade or force a reinstall. If you encounter package conflicts, try these commands:  

> pip install --upgrade --force-reinstall <package>
> pip install -I <package>  # Short for --ignore-installed
> pip install --ignore-installed <package>

Step 2: Run the `app_api.py`

Step 4: API Testing
Test your API using `curl` or any HTTP client (e.g., Postman, Thunder Client). Example:  

Checking API (/health)

 > curl -X GET http://127.0.0.1:5000/health

 > curl -X GET http://127.0.0.1:5000/stats

 > curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Age\": 20.0, \"Annual_Premium\": 30000.0, \"Policy_Sales_Channel\": 100.0, \"Gender_Male\": True, \"Vehicle_Age_< 1 Year\": False,  \"Vehicle_Age_> 2 Years\": False,  \"Vehicle_Damage_Yes\": False,  \"Driving_License_1\": True,  \"Previously_Insured_1\": False}"
    

# How to Modify
---

In frontend there are these two variables, which should be the location of the backend:

 ip_api = "127.0.0.1"

 port_api = "5000"

This code is at the bottom of the backend:

 uvicorn.run(app, host="127.0.0.1", port=5000)

Note ip is "127.0.0.1" for host computer, "0.0.0.0" for docker, while that of the server if running on a server.

In the predictor:

 > predictions = model.predict(ds_new)
 
 > #predictions = model.predict(ds_new[:, 0:6])

In the transformer:

 > #ds_new = pca.transform(dataset)

 > #return ds_new

 > return dataset


# Team members:
---

Nikita Korenko

Elizabeth Gould

Sofiia Mylnikova

Gibert Elena
