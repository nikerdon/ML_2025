# ML_2025
---

Last Edit: 12.04.2025 by Elizabeth Gould


# Status
---

* Done -- Preprocessing, Backend, Frontend, Training
* In Progress -- Testing, Documentation
* To Do -- Docker
* Persistent -- linking pieces

* Other (optional) -- Translations

# How to Use
---

Backend: python app_api.py

 * curl -X GET http://127.0.0.1:5000/health
 * curl -X GET http://127.0.0.1:5000/stats
 * curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Age\": 20.0, \"Annual_Premium\": 30000.0, \"Policy_Sales_Channel\": 100.0, \"Gender_Male\": True, \"Vehicle_Age_< 1 Year\": False,  \"Vehicle_Age_> 2 Years\": False,  \"Vehicle_Damage_Yes\": False,  \"Driving_License_1\": True,  \"Previously_Insured_1\": False}"

Frontend: streamlit run streamlit_app.py


# How to Modify
---

In frontend there are these two variables, which should be the location of the backend:

 * ip_api = "127.0.0.1"
 * port_api = "5000"

This code is at the bottom of the backend:

 * uvicorn.run(app, host="127.0.0.1", port=5000)

Note ip is "127.0.0.1" for host computer, "0.0.0.0" for docker, while that of the server if running on a server.


Other modifications for model-specific considerations are in the comments. The same transformations which were applied to the training data need to be applied to the test data.

In the predictor:

 * #predictions = model.predict(ds_new)
 * predictions = model.predict(ds_new[:, 0:6])

In the transformer:

 * ds_new = pca.transform(dataset)
 * return ds_new
 * #return dataset


# Preprocessing
---

I am going to let others move the comments here.

# Training and Testing
---

Three potential starting points:

    preprocessing.ipynb 

    training.ipynb

    train_example.ipynb (titanic)

Data here: train_short.csv, test_short.csv, X_train_phy.pkl, X_train_pca.pkl, y_train.pkl, X_test_phy.pkl, X_test_pca.pkl, y_test.pkl

scoring : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter Варианты: ‘balanced_accuracy’, ‘f1’, ‘precision’ -- Нужно смотреть, но не знаю, как пользоваться. Он сказал, ‘precision’  важнее 'recall', но чтобы делать свою оценку, нам нужно ее придумать и спрограммировать. 
(‘roc_auc’ тоже существует.)

n_jobs наверно полезно

Classifiers:
1. Decision Tree -- Trees tend to want 20-50 minimum samples and to go very deep. 
2. Random Forest
3. K-Nearest Neighbors (KNN) -- kNN is not very good for a large dataset due to the fact that it is not trained, but runs after the fact, on every new data point. In principle, this model can be useful, but would require substantial thinning or clusterization and averaging to get good results.
4. Support Vector Machine (SVM) -- This is the non-linear version of linear classification. Attempts to run this model with n_jobs = 8 did not finish even after 12 hours. It is possible that a linear classifier without a non-linear kernel can give decent results.
5. Naive Bayes: Naive Bayes seems like a good choice due to the large number of datapoints and categorical nature of most of the data.
6. other ensembles : https://scikit-learn.org/stable/modules/ensemble.html
Bagging seem like the best choice.

Мне кажется, лучше не пробовать boosting.

Decision Tree Best Params: {'max_depth': 10, 'min_samples_split': 15} Time:  3825.9848759174347  sec or  63.76641459862391 min. Не мощный компьютер, без распараллеливания.

Деревья и KNN ~ 17 мин. SVM сейчас хочет много времени. Попробую распараллеливание. Я могу послить готовые модель для проверки / тестирования. Было бы лучше, если я знала то, что вы хотели пробовать. Там разные деревья. Сейчас делаю KNN. Я пробую лес ночью. Я не знаю, как хорошо они. Деревья хочет 50 min members и так глубоко как можно. Я не могу послать kNN, потому что размер похоже на размер данных. Но они не очень медленные. 'precision' хочет больше соседов, а других меньше. Сейчас лес. Лес: меньше 2 часа с n_jobs = 8. SVM ещё не закончится.

# Backend and Frontend
---

Backend for prediction from model. Code modified from example in class. I have not added features beyond the example code. Note that my edits for the comments are in English, while the original are in Russian.

 1. Required libraries:
  > pip install fastapi uvicorn pydantic scikit-learn pandas
 Note: You may need upgrade and force reinstall. When I installed this, I had a problem with the packages.

  > pip install --upgrade --force-reinstall <package>
  > pip install -I <package>
  > pip install --ignore-installed <package>

 2. To run the app, use the following command: 
  > python app_api.py
 3. URLs for accessing the code:

  > curl -X GET http://127.0.0.1:5000/health
  > curl -X GET http://127.0.0.1:5000/stats
  > curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Age\": 20.0, \"Annual_Premium\": 30000.0, \"Policy_Sales_Channel\": 100.0, \"Gender_Male\": True, \"Vehicle_Age_< 1 Year\": False,  \"Vehicle_Age_> 2 Years\": False,  \"Vehicle_Damage_Yes\": False,  \"Driving_License_1\": True,  \"Previously_Insured_1\": False}"

The code works now with the Gaussian Bayes model. I checked here to make sure the input properly changes the test data as well. You will need three pickles for the code to work: scaler.pkl, pca.pkl, model.pkl. You will need to adjust the code, based on whether or not the model uses PCA. For rescaling data, three posibilities exist: standard_scale object, csv file with data of mean, then std, or just adding the data to the .py file. The third variant is not recommended, while the first is implemented. Potential modifications are in the comments, as well as modifications for testing. You just need to uncomment them, while commenting out the other version.

---

Frontend streamlit app for ML 2025 project 1

Based on (uses as template) -- 
https://github.com/Koldim2001/test_api/blob/microservices-example/streamlit-service/streamlit_app.py

Run code: streamlit run streamlit_app.py

Everything here is basic. It should limit the responses to be within the acceptable range, with a waring when you get beyond the training data range. I have a slight change in the setup of how it checks if a response is valid because I was contemplating adding in functionality where it hides unnecesary input. Note that the input text boxes will not recognize a negative number or decimal value as a valid input. I believe I can fix that, but I actually want this for most of the input.


