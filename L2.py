import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv(
    "/Users/moodgame/last/ML/learning/Automobile_data.csv", na_values="?"
)
clean_data = data.dropna(subset=["price"])

num_cols = ["engine-size", "horsepower"]
cat_cols = ["make"]

X = clean_data[num_cols + cat_cols]
y = clean_data["price"]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Ridge (alpha=10.0)": Ridge(alpha=10.0),
}

print("=== СРАВНЕНИЕ НА 5-ФОЛДОВОЙ КРОСС-ВАЛИДАЦИИ ===")
for name, reg in models.items():
    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", reg)]
    )
    scores = -cross_val_score(
        pipe, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )
    print(f"{name:<20} | MAE: {scores.mean():.2f}$ (± {scores.std():.2f}$)")

pipe_lr = Pipeline(
    [("preprocessor", preprocessor), ("regressor", LinearRegression())]
).fit(X, y)
pipe_ridge = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))]
).fit(X, y)

feature_names = pipe_lr.named_steps["preprocessor"].get_feature_names_out()

weights_df = pd.DataFrame({
    "feature": feature_names,
    "Linear_weight": pipe_lr.named_steps["regressor"].coef_,
    "Ridge_weight": pipe_ridge.named_steps["regressor"].coef_,
})

weights_df["abs_lr"] = weights_df["Linear_weight"].abs()
weights_df = weights_df.sort_values(by="abs_lr", ascending=False).drop(
    columns=["abs_lr"]
)

print("\n=== ТОП-10 ПРИЗНАКОВ ПО ВЕСУ ===")
print(weights_df.head(10).to_string(index=False))
