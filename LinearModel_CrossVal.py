import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv(
    "/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values="?"
)
clean_data_price = data.dropna(subset=["price"])


numeric_features = ["engine-size", "horsepower"]
categorical_features = ["make"]
features = numeric_features + categorical_features


X = clean_data_price[features]
y = clean_data_price["price"]


check_corr = ["engine-size", "horsepower"]
corr_matrix = clean_data_price[check_corr].corr()

print(corr_matrix.round(2))


preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

print("=" * 50)
print(f"MAE на каждом из 5 фолдов ($):")
for i, score in enumerate(scores, 1):
    print(f"  Фолд {i}: {score:.2f}$")

print("-" * 50)
print(f"Средний MAE:               {scores.mean():.2f}$")
print(f"Стандартное отклонение (±): {scores.std():.2f}$")
print("=" * 50)
