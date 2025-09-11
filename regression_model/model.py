import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)

# Regression libraries
from catboost import CatBoostRegressor


def trainModel(df, obs, target="price"):
    # Split into train/test
    X = df.drop(columns=[target])
    y = df[target]
    # Apply log1p transform to target
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=123
    )

    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123)

    # Fit on training set
    model.fit(X_train, y_train_log)

    # Predict on train and test sets
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    # Cross-validation on training set
    cv = KFold(n_splits=10, shuffle=True, random_state=123)
    y_cv_pred_log = cross_val_predict(model, X_train, y_train_log, cv=cv)

    # Get regression results
    train_results = regResults(obs, "CatBoost", y_train_log, y_train_pred_log)
    test_results = regResults(obs, "CatBoost", y_test_log, y_test_pred_log)
    cv_results = regResults(obs, "CatBoost CV", y_train_log, y_cv_pred_log)

    # Prepare results dictionary
    combined_result = {
        "Model": "CatBoost",
        "Features": obs,
        "Runtime (min)": None,
        **{
            k + " (Train)": v
            for k, v in train_results.items()
            if k not in ["Model", "Observations", "Features"]
        },
        **{
            k + " (CV)": v
            for k, v in cv_results.items()
            if k not in ["Model", "Observations", "Features"]
        },
        **{
            k + " (Test)": v
            for k, v in test_results.items()
            if k not in ["Model", "Observations", "Features"]
        },
    }

    pd.DataFrame([combined_result]).to_csv("regression_model/results.csv", index=False)

    return model


def regResults(features, model_name, y_true_log, y_pred_log):
    """
    Compute regression metrics (R², MAE, RMSE) on the original scale
    by converting log-transformed predictions and targets back to price.

    Parameters:
        features (str): Features used.
        model_name (str): Name of the model.
        y_true_log (array-like): Log-transformed true target values.
        y_pred_log (array-like): Log-transformed predicted values.

    Returns:
        dict: Regression metrics on the original price scale.
    """
    # Clip log values to avoid overflow in expm1
    max_log = 20  # expm1(20) ~ 4.85 million
    y_true_log = np.clip(y_true_log, None, max_log)
    y_pred_log = np.clip(y_pred_log, None, max_log)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "Features": features,
        "Model": model_name,
        "R² Score": r2,
        "MAE": mae,
        "RMSE": rmse,
    }
