import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv(
    "/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values="?"
)
clean_data = data.dropna(subset=["price"])
y = clean_data["price"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

feature_sets = {
    "1. База (make + engine + hp)": (
        ["engine-size", "horsepower"],
        ["make"],
    ),
    "2. + масса и габариты (curb-weight, width)": (
        ["engine-size", "horsepower", "curb-weight", "width"],
        ["make"],
    ),
    "3. + кузов и привод (body-style, drive-wheels)": (
        ["engine-size", "horsepower", "curb-weight", "width"],
        ["make", "body-style", "drive-wheels"],
    ),
    "4. + расход и топливо (city-mpg, fuel-type)": (
        ["engine-size", "horsepower", "curb-weight", "width", "city-mpg"],
        ["make", "body-style", "drive-wheels", "fuel-type"],
    ),
}

for name, (num_cols, cat_cols) in feature_sets.items():
    X = clean_data[num_cols + cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    scores = -cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )
    print(f"{name:<48} | MAE: {scores.mean():.2f}$ (± {scores.std():.2f}$)")
