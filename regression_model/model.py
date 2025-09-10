
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split

# Regression libraries
from catboost import CatBoostRegressor

def trainModel(df, target="price"):
    # Split into train/test
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123)

    # Fit on training set
    model.fit(X_train, y_train)

    return model

