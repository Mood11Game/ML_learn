import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression

import joblib


data = pd.read_csv("/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values='?')


clean_miss_price = data.dropna(subset=['price'])


X = clean_miss_price.drop(columns=['price'])
y = clean_miss_price['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train[['engine-size']], y_train)

preds_lr = lr.predict(X_test[['engine-size']])
mae_lr = mean_absolute_error(y_test, preds_lr)

print(f"mae LinerRegression: {mae_lr}")
