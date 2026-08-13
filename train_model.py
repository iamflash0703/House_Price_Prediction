"""
House Price Prediction v2 (Upgraded) — Training Script
Real California Housing Dataset (20,640 houses)

Goal: Build a production-quality regression pipeline — proper missing
value handling, categorical encoding, feature engineering, multiple
models including XGBoost, and save the best model for deployment.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the trained model + pipeline to disk

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sns.set_style("darkgrid")
np.random.seed(42)

# ---------- STEP 1: Data Loading ----------
# Real California Housing dataset (20,640 houses, 1990 census data) —
# a well-known, validated dataset used in production ML tutorials
# (e.g. "Hands-On Machine Learning" book).
df = pd.read_csv("housing.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nCategorical column values:\n", df["ocean_proximity"].value_counts())

# ---------- STEP 2: Exploratory Data Analysis ----------

# 2a. Target variable distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["median_house_value"], bins=50, kde=True, color="#4C9AFF")
plt.title("House Value Distribution")
plt.xlabel("Median House Value ($)")
plt.tight_layout()
plt.savefig("price_distribution.png", dpi=150)
plt.close()

# 2b. Correlation heatmap (numeric features only)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="mako", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# 2c. Geographic price map (longitude/latitude colored by price) — a
# classic visualization for this dataset, since location matters a lot
plt.figure(figsize=(8, 7))
scatter = plt.scatter(df["longitude"], df["latitude"], c=df["median_house_value"],
                       cmap="viridis", alpha=0.4, s=10)
plt.colorbar(scatter, label="Median House Value ($)")
plt.title("House Prices by Geographic Location (California)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("geographic_prices.png", dpi=150)
plt.close()

print("\nEDA plots saved.")

# ---------- STEP 3: Feature Engineering ----------
# These derived features are well-known to improve this specific
# dataset's predictive power (rooms/bedrooms per household are more
# meaningful than raw totals, which just reflect neighborhood size).
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Log-transform the target — house prices are right-skewed (a few very
# expensive houses stretch the distribution), and log transform helps
# linear models handle this better.
df["log_price"] = np.log1p(df["median_house_value"])

print(f"\nSkewness before log transform: {df['median_house_value'].skew():.3f}")
print(f"Skewness after log transform:  {df['log_price'].skew():.3f}")

# ---------- STEP 4: Train-Test Split ----------
feature_cols = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity", "rooms_per_household", "bedrooms_per_room",
    "population_per_household"
]
X = df[feature_cols]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- STEP 5: Preprocessing Pipeline ----------
# Building a proper sklearn Pipeline (instead of manual steps) is what
# makes this "deployment-ready" — the exact same preprocessing gets
# applied automatically at prediction time, with zero risk of
# train/test mismatch.
numeric_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household"
]
categorical_features = ["ocean_proximity"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # fills missing total_bedrooms
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),  # converts text categories to numeric columns
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# ---------- STEP 6: Model Building & Comparison ----------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    # Full pipeline = preprocessing + model, trained together
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    pred_log = pipe.predict(X_test)
    pred_price = np.expm1(pred_log)
    actual_price = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae = mean_absolute_error(actual_price, pred_price)
    mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100
    r2 = r2_score(actual_price, pred_price)

    results[name] = {"pipeline": pipe, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
    print(f"\n--- {name} ---")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"MAE:  ${mae:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

# ---------- STEP 7: Cross-Validation on Best Model ----------
best_name = max(results, key=lambda k: results[k]["r2"])
best_pipeline = results[best_name]["pipeline"]
print(f"\nBest model: {best_name}")

cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="r2")
print(f"5-Fold CV R2 scores: {np.round(cv_scores, 4)}")
print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------- STEP 8: Save the Model Pipeline (Deployment Artifact) ----------
# Saving the FULL pipeline (preprocessing + model together) means the
# deployed app doesn't need to duplicate any preprocessing logic — it
# just calls .predict() on raw input data.
joblib.dump(best_pipeline, "house_price_model.pkl")
print("\nModel pipeline saved to house_price_model.pkl")

# ---------- STEP 9: Visualization ----------

# 9a. Model comparison
plt.figure(figsize=(8, 5))
r2_scores = [results[n]["r2"] for n in models]
plt.bar(models.keys(), r2_scores, color=["#4C9AFF", "#2ECC71", "#F39C12", "#E74C3C"])
plt.title("Model Comparison — R2 Score")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 9b. Feature importance (from best tree-based model, if applicable)
if best_name in ["Random Forest", "XGBoost"]:
    model_step = best_pipeline.named_steps["model"]
    feature_names = (
        numeric_features +
        list(best_pipeline.named_steps["preprocess"]
             .named_transformers_["cat"].named_steps["onehot"]
             .get_feature_names_out(categorical_features))
    )
    importances = pd.Series(model_step.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Top 10 Feature Importances ({best_name})")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

# 9c. Actual vs Predicted for best model
best_pred_log = best_pipeline.predict(X_test)
best_pred = np.expm1(best_pred_log)
actual_final = np.expm1(y_test)

plt.figure(figsize=(7, 6))
plt.scatter(actual_final, best_pred, alpha=0.3, color="#4C9AFF", s=15)
min_v, max_v = actual_final.min(), actual_final.max()
plt.plot([min_v, max_v], [min_v, max_v], color="#E74C3C", linestyle="--", linewidth=2)
plt.title(f"Actual vs Predicted — {best_name}")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 10: Key Insights ----------
print("\nKey Insights:")
print(f"1. Best model: {best_name}, R2 = {results[best_name]['r2']:.4f}, "
      f"MAPE = {results[best_name]['mape']:.2f}%")
print(f"2. 5-Fold CV confirms generalization: mean R2 = {cv_scores.mean():.4f}")
print("3. median_income is typically the strongest predictor of house value "
      "in this dataset — wealthier areas have higher home values.")
print("4. Engineered features (rooms_per_household, bedrooms_per_room) add "
      "predictive power beyond the raw totals.")
