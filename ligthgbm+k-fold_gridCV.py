import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

data = pd.read_csv("dataset/train.csv")
X = data.iloc[:, :55]
y = data.iloc[:, 55:]

fixed_params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "bagging_freq": 5,
    "random_state": 42,
    "device": "gpu",              
    "gpu_platform_id": 0,          
    "gpu_device_id": 0             
}

param_grid = {
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 5, 6, 7, 8],
    'num_leaves': [24, 30, 36, 42],
    'min_data_in_leaf': [40, 50, 60, 70, 80],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'lambda_l1': [0.0, 0.1, 0.3, 0.5],
    'lambda_l2': [0.0, 0.1, 0.3, 0.5],
}

scorers = {
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'R2': make_scorer(r2_score)
}

kf = KFold(n_splits=12, shuffle=True, random_state=42)
results = []

for target_col in y.columns:
    print(f"\n🚀 Grid searching for {target_col} with GPU...")

    Y = y[target_col]
    model = lgb.LGBMRegressor(**fixed_params)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorers,
        refit='MAE',
        cv=kf,
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X, Y)
    best_model = grid_search.best_estimator_
    pred = best_model.predict(X)
    mae = mean_absolute_error(Y, pred)
    r2 = r2_score(Y, pred)

    print(f" {target_col} → MAE: {mae:.4f} | R²: {r2:.4f}")

    results.append({
        'BlendProperty': target_col,
        'MAE': mae,
        'R2 Score': r2,
        'Best Params': grid_search.best_params_
    })

results_table = pd.DataFrame(results)
print("grid search results:")
print(results_table)
