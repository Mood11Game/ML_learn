import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import ConstantMeanRegressor as cmr
import MakeMeanRegressin as mmr
import joblib

#########
#Prepare data
###########


data = pd.read_csv("/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values='?')

print(data['make'].isna().sum())

check_miss = data['price'].isna().sum()

#9, 44, 45, 129
#miss_price = data[data['price'].isna()]

#       
# mean_price = data['price'].mean() 13207.129353233831 
# median_price = data['price'].median() 10295.0

clean_miss_price = data.dropna(subset=['price'])


################
# Separate data X Y and train-test
################

X = clean_miss_price.drop(columns=['price'])
y = clean_miss_price['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Обучающая выборка: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Тестовая выборка:  X_test  = {X_test.shape}, y_test  = {y_test.shape}")


################################
## train and MAE DUMMY
################################

print(50*'#')
model = cmr.ConstantMeanRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Средняя цена в train: {model.mean_value_:.2f}")
print(f"MAE константного бейзлайна: {mae:.2f}")

print(50*'#')

model1 = mmr.MakeMeanRegressin()
model1.fit(X_train, y_train)

# Вызываем метод у model1 и сохраняем в preds1
preds1 = model1.predict(X_test)

# Считаем MAE именно для preds1
mae1 = mean_absolute_error(y_test, preds1)

print(f"Средняя цена в train: {model1.global_mean_:.2f}")
print(f"MAE бейзлайна по марке (make): {mae1:.2f}")

result = pd.DataFrame({
    'active_prise' : y_test.values,
    'predicted_make' : preds1,
    'error' : y_test.values - preds1
})

result.to_csv("predictions.csv", index=False)
joblib.dump(model1, "make_model.joblib")



################################
## plot
################################

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # -------------------------------------------------------------
# # График 1: Реальные цены против предсказанных
# # -------------------------------------------------------------
# axes[0].scatter(y_test, preds, color='blue', alpha=0.7, label='Предсказания бейзлайна')
# # Идеальная линия (как предсказывала бы совершенная модель)
# axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Идеальная модель (y = x)')
# axes[0].set_title('Реальные цены vs Предсказанные')
# axes[0].set_xlabel('Реальная цена ($)')
# axes[0].set_ylabel('Предсказанная цена ($)')
# axes[0].legend()
# axes[0].grid(True, linestyle=':', alpha=0.6)

# # -------------------------------------------------------------
# # График 2: График остатков (ошибок)
# # -------------------------------------------------------------
# residuals = y_test - preds
# axes[1].scatter(y_test, residuals, color='purple', alpha=0.7)
# axes[1].axhline(0, color='red', linestyle='--', label='Нулевая ошибка')
# axes[1].set_title('График остатков (Ошибки модели)')
# axes[1].set_xlabel('Реальная цена ($)')
# axes[1].set_ylabel('Ошибка (Реальная - Предсказанная, $)')
# axes[1].legend()
# axes[1].grid(True, linestyle=':', alpha=0.6)

# plt.tight_layout()
# plt.show()


