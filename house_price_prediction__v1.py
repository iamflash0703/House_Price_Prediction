"""
Major Project 2: House Price Prediction
Data Science Major Projects (Beginner, 1-2 month track)

Goal: Predict house prices using MULTIPLE features (unlike Project 2's
single-feature regression), with feature engineering and log
transformation, comparing Linear, Ridge, and Random Forest regression.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style("darkgrid")
np.random.seed(42)

# ---------- STEP 1: Data Preparation ----------
# Real datasets (Ames Housing / California Housing) aren't reachable in
# this environment (external download blocked), so we generate a
# realistic synthetic multi-feature housing dataset instead - this time
# with 6 features (not just 1, unlike the earlier Salary project), so
# we can practice feature engineering and multi-variable regression.
n_samples = 500

sqft = np.random.normal(1800, 600, n_samples).clip(500, 4500)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
distance_to_city = np.random.uniform(1, 30, n_samples)  # km from city center
neighborhood_quality = np.random.uniform(1, 10, n_samples)  # 1-10 score

# Price formula: realistic relationship + noise
# Bigger house, more bedrooms/bathrooms, better neighborhood -> higher price
# Older house, farther from city -> lower price
price = (
    50000
    + sqft * 120
    + bedrooms * 8000
    + bathrooms * 6000
    - age * 800
    - distance_to_city * 1500
    + neighborhood_quality * 10000
    + np.random.normal(0, 20000, n_samples)  # random noise
)
price = price.clip(50000, None)  # no negative prices

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age,
    "distance_to_city_km": distance_to_city,
    "neighborhood_quality": neighborhood_quality,
    "price": price,
})

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ---------- STEP 2: EDA ----------

# 2a. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="mako", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# 2b. Price distribution (before log transform) - check for skewness
plt.figure(figsize=(7, 5))
sns.histplot(df["price"], bins=40, kde=True, color="#4C9AFF")
plt.title("House Price Distribution (Original)")
plt.xlabel("Price")
plt.tight_layout()
plt.savefig("price_distribution.png", dpi=150)
plt.close()

# 2c. Price vs sqft scatter, colored by neighborhood quality
plt.figure(figsize=(7, 5.5))
scatter = plt.scatter(df["sqft"], df["price"], c=df["neighborhood_quality"],
                       cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Neighborhood Quality")
plt.title("Price vs Square Footage (colored by Neighborhood Quality)")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("price_vs_sqft.png", dpi=150)
plt.close()

# ---------- STEP 3: Feature Engineering ----------
# 3a. Log-transform price - reduces skewness in the target variable,
# which often helps linear models perform better and make more
# stable predictions (common technique for right-skewed price/income data)
df["log_price"] = np.log1p(df["price"])  # log1p = log(1+x), safe for any positive value

# 3b. New engineered feature: price per square foot potential proxy -
# total rooms (bedrooms + bathrooms combined)
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

print("\nSkewness before log transform:", df["price"].skew())
print("Skewness after log transform:", df["log_price"].skew())

# ---------- STEP 4: Model Building ----------
feature_cols = ["sqft", "bedrooms", "bathrooms", "age_years",
                 "distance_to_city_km", "neighborhood_quality", "total_rooms"]
X = df[feature_cols]
y = df["log_price"]  # predicting log price (will convert back later for interpretation)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Random Forest":
        model.fit(X_train, y_train)          # trees don't need scaling
        pred_log = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        pred_log = model.predict(X_test_scaled)

    # Convert predictions back from log scale to actual price scale
    pred_price = np.expm1(pred_log)
    actual_price = np.expm1(y_test)

    mse = mean_squared_error(actual_price, pred_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_price, pred_price)
    r2 = r2_score(actual_price, pred_price)

    results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "pred": pred_price}
    print(f"\n--- {name} ---")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# ---------- STEP 5: Feature Importance (Random Forest) ----------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nFeature Importances (Random Forest):\n", importances)

# ---------- STEP 6: Visualization ----------

# 6a. Model comparison (R2 scores)
plt.figure(figsize=(7, 5))
r2_scores = [results[n]["r2"] for n in models]
plt.bar(models.keys(), r2_scores, color=["#4C9AFF", "#2ECC71", "#F39C12"])
plt.title("Model Comparison — R2 Score")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 6b. Feature importance
plt.figure(figsize=(8, 5.5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# 6c. Actual vs Predicted for best model
best_name = max(results, key=lambda k: results[k]["r2"])
best_pred = results[best_name]["pred"]
actual_price_final = np.expm1(y_test)

plt.figure(figsize=(7, 6))
plt.scatter(actual_price_final, best_pred, alpha=0.6, color="#4C9AFF")
min_v, max_v = actual_price_final.min(), actual_price_final.max()
plt.plot([min_v, max_v], [min_v, max_v], color="#E74C3C", linestyle="--", linewidth=2)
plt.title(f"Actual vs Predicted Price — {best_name}")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 7: Key Insights ----------
print("\nKey Insights:")
print(f"1. Best model: {best_name} with R2 = {results[best_name]['r2']:.4f}")
print(f"2. Most important feature for price: '{importances.index[0]}'")
print("3. Log-transformation is standard practice for real housing data (which is "
      "typically right-skewed) - in this synthetic dataset the price distribution "
      "was already fairly symmetric, so the technique is demonstrated but had "
      "limited impact here. On real-world data it usually helps more.")
