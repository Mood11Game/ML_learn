import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import joblib


data = pd.read_csv("/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values='?')

# print(data[data['horsepower'].isna()])


clean_data_price = data.dropna(subset=['price'])

features = ["engine-size", "curb-weight", "horsepower"]

X = clean_data_price[features]
y = clean_data_price['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

imputer = SimpleImputer(strategy="median")

X_train_imputer = imputer.fit_transform(X_train)
X_test_imputer = imputer.fit_transform(X_test)

multi_lr = LinearRegression()
multi_lr.fit(X_train_imputer, y_train)

preds_lr = multi_lr.predict(X_test_imputer)
mae_lr = mean_absolute_error(preds_lr, y_test)

print(f"MAE константы:                     8262.03$")
print(f"MAE по маркам:                     3703.80$")
print(f"MAE линейной (только engine-size): 4176.00$")
print(f"MAE линейной (3 признака):         {mae_lr:.2f}$")

#3681.31