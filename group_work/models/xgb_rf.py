import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from feature_selection import feature_select_pearson
from pathlib import Path


def grid_search_cv(x_train, y_train):
    # The parameter space
    parameter_space = {
        'learning_rate': [1.0, 1.2],
        'max_depth': [5, 6],
        'min_child_weight': [4, 5],
        'n_estimators': [80]
    }
    # do the grid search with cross validation
    print("Tuning hyper-parameters by gridsearch")
    model = xgb.XGBRFRegressor(objective="reg:squarederror", n_jobs=15, seed=42)
    grid = GridSearchCV(model, parameter_space, cv=2, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)

    # get the best parameters on the sample of training set
    print("The best parameters are:")
    print(grid.best_params_)
    return grid.best_params_


def xgbrf_filter_default(train, test):
    print('XGBoostRF with filter method to do feature selection with default parameters')
    train, test = feature_select_pearson(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = xgb.XGBRFRegressor(objective="reg:squarederror", seed=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE on validation set is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


def xgbrf_filter(train, test, best_params):
    print('XGBoost with filter method to do feature selection with best parameters')
    train, test = feature_select_pearson(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = xgb.XGBRFRegressor(**best_params, objective="reg:squarederror", seed=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


data_folder = Path("../preprocess/")
train = pd.read_csv(data_folder / 'new_train.csv')
test = pd.read_csv(data_folder / 'new_test.csv')

# user a small sample of training set to find the best parameters by gridsearch
train_sample = pd.read_csv(data_folder / 'new_train_30perc.csv')
# best_params = grid_search_cv(train_sample.loc[:, train_sample.columns != 'card_id'], train_sample['target'])

saved_best_params = {'learning_rate': 1.0, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 80}

xgbrf_filter_default(train, test)
xgbrf_filter(train, test, saved_best_params)
