import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# 1. Загрузка
data = pd.read_csv("/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values="?")
clean_data = data.dropna(subset=["price"])

# 2. Финальный набор признаков
num_cols = ["engine-size", "horsepower", "curb-weight", "width", "city-mpg"]
cat_cols = ["make", "body-style", "drive-wheels", "fuel-type"]

X = clean_data[num_cols + cat_cols]
y = clean_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

final_model.fit(X_train, y_train)

preds = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, preds)

print(f"Финальный MAE на отложенном тесте: {final_mae:.2f}$")


joblib.dump(final_model, "car_price_rf_model.joblib")
print("Модель сохранена в файл 'car_price_rf_model.joblib'")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, preds - y_test, alpha=0.7, color='#d62728', edgecolors='black')
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Реальная цена автомобиля ($)")
plt.ylabel("Ошибка модели (Предсказание - Реальность)")
plt.title("График остатков: Случайный лес")
plt.grid(True, alpha=0.3)
plt.show()
