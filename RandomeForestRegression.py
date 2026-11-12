import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv(
    "/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values="?"
)
clean_data = data.dropna(subset=["price"])

num_cols = ["engine-size", "horsepower", "curb-weight", "width", "city-mpg"]
cat_cols = ["make", "body-style", "drive-wheels", "fuel-type"]

X = clean_data[num_cols + cat_cols]
y = clean_data["price"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LinearRegression (Max features)": LinearRegression(),
    "RandomForest (Max features)": RandomForestRegressor(random_state=42)
}

print("=== СРАВНЕНИЕ АЛГОРИТМОВ НА МАКСИМАЛЬНЫХ ПРИЗНАКАХ ===")
for name, model_algo in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model_algo),
    ])
    scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    print(f"{name:<35} | MAE: {scores.mean():.2f}$ (± {scores.std():.2f}$)")
