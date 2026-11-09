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
clean_data = data.dropna(subset=["price"])

y = clean_data["price"]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

experiments = {
    "engine-size + horsepower": ["engine-size", "horsepower"],
    "только engine-size": ["engine-size"],
    "только horsepower": ["horsepower"],
}

for name, num_cols in experiments.items():
    X = clean_data[num_cols + ["make"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["make"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    scores = -cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )
    print(f"{name:<25} | MAE: {scores.mean():.2f}$ (± {scores.std():.2f}$)")
