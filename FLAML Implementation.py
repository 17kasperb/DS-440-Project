# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from flaml import AutoML
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_log_error, mean_squared_error, r2_score
# Classification with FLAML
# %%
from sklearn.datasets import load_iris
dataset = load_iris()
dataset
# %%
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
automl_clf = AutoML()
automl_clf.fit(x_train, y_train, task="classification")
# %%
y_pred = automl_clf.predict(x_test)
accuracy_score(y_test, y_pred)
# %%
y_pred

# Regression with FLAML
# %%
dataset1 = load_boston()
# %%
x = dataset1.data
y = dataset1.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2, random_state = 100)
automl_reg = AutoML()
automl_reg.fit(x_train, y_train, task="regression")
# %%
y_pred = automl_reg.predict(x_test)
print('MEV :', max_error(y_test, y_pred))
print('MEAV :', mean_absolute_error(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))
print("MSLE :", mean_squared_log_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# Time Series with FLAML
# %%
# pip install flaml[forecast]
X_train = np.arange('2014-01', '2021-01', dtype='datetime64[M]')
y_train = np.random.random(size=72)
automl = AutoML()
automl.fit(X_train=X_train[:72],  # a single column of timestamp
           y_train=y_train,  # value for each timestamp
           period=12,  # time horizon to forecast, e.g., 12 months
           task='forecast', time_budget=15,  # time budget in seconds
           eval_method="holdout",
           log_file_name="forecast.log",)
print(automl.predict(X_train[72:]))
