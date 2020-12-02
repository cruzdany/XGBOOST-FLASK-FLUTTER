import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv(‘kc_house_data.csv’)
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,16,17,18,19]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
traindf, testdf = train_test_split(X_train, test_size = 0.2)
res=xgb.fit(X_train,y_train)
import joblib
joblib.dump(res, "model.pkl")
cols=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view",
"condition","grade","sqft_above","sqft_basement","lat","long","sqft_living15","sqft_lot15"]
joblib.dump(cols, "model_cols.pkl")


Ultimamilla1$