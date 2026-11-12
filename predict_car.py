import pandas as pd
import joblib


model_path = "/Users/moodgame/last/ML/learning/car_price_rf_model.joblib"
loaded_model = joblib.load(model_path)


new_car = pd.DataFrame([{
    "engine-size": 164,      # объем двигателя
    "horsepower": 121,       # лошадиные силы
    "curb-weight": 2756,     # снаряженная масса
    "width": 66.8,           # ширина
    "city-mpg": 21,          # расход в городе
    "make": "bmw",           # марка
    "body-style": "sedan",   # тип кузова
    "drive-wheels": "rwd",   # привод (задний)
    "fuel-type": "gas"       # тип топлива (бензин)
}])

predicted_price = loaded_model.predict(new_car)

print(f"Введенные данные:\n{new_car.iloc[0]}\n")
print("-" * 30)
print(f"Предсказанная стоимость: {predicted_price[0]:.2f}$")
