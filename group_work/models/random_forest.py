import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from feature_selection import feature_select_pearson, random_forest_wrapper
from pathlib import Path


def grid_search_cv(x_train, y_train):
    # The parameter space
    parameter_space = {
        "min_samples_leaf": [30, 31],
        "min_samples_split": [2, 3],
        "max_depth": [9, 10],
        "max_features": [80]
    }
    # do the grid search with cross validation
    print("Tuning hyper-parameters by gridsearch")
    model = RandomForestRegressor(
        criterion="squared_error",
        n_jobs=15,
        random_state=42)
    grid = GridSearchCV(model, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(x_train, y_train)

    # get the best parameters on the sample of training set
    print("The best parameters are:")
    print(grid.best_params_)
    return grid.best_params_


def rf_filter_default(train, test):
    print('Randomforest with filter method to do feature selection with default parameters')
    train, test = feature_select_pearson(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = RandomForestRegressor(criterion="squared_error",
                                n_jobs=15,
                                random_state=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE on validation set is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


def rf_wrapper_default(train, test):
    print('Randomforest with wrapper method to do feature selection with default parameters')
    train, test = random_forest_wrapper(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = RandomForestRegressor(criterion="squared_error",
                                n_jobs=15,
                                random_state=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE on validation set is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


def rf_filter(train, test, best_params):
    print('Randomforest with filter method to do feature selection with best parameters')
    train, test = feature_select_pearson(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = RandomForestRegressor(**best_params, criterion="squared_error",
                                n_jobs=15,
                                random_state=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE on validation set is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    result_folder = Path("../result/")
    result_folder.mkdir(parents=True, exist_ok=True)

    # add predict target to testing set
    x_test['predict_target'] = y_pred
    x_test[['card_id', 'predict_target']].to_csv(result_folder / 'randomforest_filter.csv', index=False)


def rf_wrapper(train, test, best_params):
    print('Randomforest with wrapper method to do feature selection with best parameters')
    train_ = train.copy()
    train, test = random_forest_wrapper(train, test)

    data_folder = Path("../preprocess/")
    kaggle_test = pd.read_csv(data_folder / 'test.csv')
    _, kaggle_test = random_forest_wrapper(train_, kaggle_test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    reg = RandomForestRegressor(**best_params, criterion="squared_error",
                                n_jobs=15,
                                random_state=42)
    reg.fit(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    y_pred = reg.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The RMSE is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    result_folder = Path("../result/")
    result_folder.mkdir(parents=True, exist_ok=True)

    # add predict target to testing set
    x_test['predict_target'] = y_pred
    x_test[['card_id', 'predict_target']].to_csv(result_folder / 'randomforest_wrapper.csv', index=False)

    kaggle_test['target'] = reg.predict(kaggle_test.loc[:, kaggle_test.columns != 'card_id'])
    kaggle_test[['card_id', 'target']].to_csv(result_folder / 'submission_randomforest_wrapper.csv', index=False)


data_folder = Path("../preprocess/")
train = pd.read_csv(data_folder / 'new_train.csv')
test = pd.read_csv(data_folder / 'new_test.csv')

# user a small sample of training set to find the best parameters by gridsearch
train_sample = pd.read_csv(data_folder / 'new_train_30perc.csv')
# best_params = grid_search_cv(train_sample.loc[:, train_sample.columns != 'card_id'], train_sample['target'])

# Use the result of the best parameters we trained to save computing time
saved_best_params = {
    'max_depth': 10,
    'max_features': 80,
    'min_samples_leaf': 31,
    'min_samples_split': 2
}

rf_filter_default(train, test)
rf_wrapper_default(train, test)
rf_filter(train, test, saved_best_params)
rf_wrapper(train, test, saved_best_params)
