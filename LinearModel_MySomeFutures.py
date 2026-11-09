import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values='?')

# print(data[data['horsepower'].isna()])


clean_data_price = data.dropna(subset=['price'])

numeric_features = ["engine-size", "horsepower"]
categorial_futures = ['make']
features = numeric_features + categorial_futures

X = clean_data_price[features]
y = clean_data_price['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),  
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorial_futures)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
mean_comb = mean_absolute_error(y_test, preds)

print(f"MAE константы:                     8262.03$")
print(f"MAE по маркам:                     3703.80$")
print(f"MAE линейной (только engine-size): 4176.00$")
print(f"MAE линейной (3 признака):         3681.31$")
print(f"MAE линейной (мои 3 признака): без кросс-валидации  {mean_comb:.2f}$")
