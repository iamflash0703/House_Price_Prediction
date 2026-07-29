# 🏠 House Price Prediction

Multi-feature regression project predicting house prices using 6 features (square footage, bedrooms, bathrooms, age, distance to city, neighborhood quality) — comparing Linear Regression, Ridge Regression, and Random Forest.

## 📌 Overview
Unlike simple single-feature regression, this project tackles a multivariate regression problem with feature engineering, log transformation, and model comparison — closer to how real-world price prediction problems are approached.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 🔍 Steps Followed
1. **Data Preparation** — Generated a realistic synthetic dataset (500 houses, 6 features) with realistic price relationships.
2. **EDA** — Correlation heatmap, price distribution, and price vs. square footage analysis.
3. **Feature Engineering** — Applied log transformation to price and created a new `total_rooms` feature.
4. **Model Building** — Trained Linear Regression, Ridge Regression, and Random Forest Regressor.
5. **Model Evaluation** — Compared RMSE, MAE, and R² Score across models.
6. **Feature Importance** — Identified which factors matter most for price using Random Forest.

## 📊 Results

| Model               | RMSE     | MAE      | R² Score |
|----------------------|----------|----------|----------|
| Linear Regression     | $27,036  | $20,792  | 0.8926   |
| Ridge Regression       | $26,924  | $20,758  | 0.8935   |
| Random Forest          | $28,731  | $22,193  | 0.8787   |

**Key finding:** Square footage is the dominant price predictor (77.7% importance), followed by neighborhood quality (11.1%).

## 📁 Files
- `house_price_prediction.py` — full commented Python script
- `*.png` — correlation heatmap, model comparison, feature importance, actual vs predicted
