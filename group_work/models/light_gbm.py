import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb
from hyperopt import hp, fmin, tpe
from feature_selection import feature_select_pearson, lightGBM_wrapper
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
from pathlib import Path


def params_append(params):
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['feature_pre_filter'] = False
    params['bagging_seed'] = 42
    return params


def param_hyperopt(train):
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')
    train_data = lgb.Dataset(train[features], train[label])

    def hyperopt_objective(params):
        params = params_append(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=2,
                     stratified=False,
                     shuffle=True,
                     metrics='rmse',
                     early_stopping_rounds=30,
                     verbose_eval=False,
                     show_stdv=False,
                     seed=42)
        return min(res['rmse-mean'])

    params_space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'reg_alpha': hp.randint('reg_alpha', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10)
    }

    # TPE search for best paramters
    params_best = fmin(
        fn=hyperopt_objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=30,
        rstate=np.random.default_rng(42))
    print('The best parameters are:')
    print(params_best)

    return params_best


def lightGBM_filter_default(train, test):
    print('LightGBM with filter method to do feature selection with default parameters')
    train, test = feature_select_pearson(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'feature_pre_filter': False,
        'bagging_seed': 42
    }

    lgb_train = lgb.Dataset(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    lgb_eval = lgb.Dataset(x_test.loc[:, x_test.columns != 'card_id'], y_test, reference=lgb_train)

    gbm = lgb.train(default_params, lgb_train, num_boost_round=1000,
                    valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=50)
    y_pred = gbm.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The score is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


def lightGBM_wrapper_default(train, test):
    print('LightGBM with wrapper method to do feature selection with default parameters')
    train, test = lightGBM_wrapper(train, test)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'feature_pre_filter': False,
        'bagging_seed': 42
    }

    lgb_train = lgb.Dataset(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    lgb_eval = lgb.Dataset(x_test.loc[:, x_test.columns != 'card_id'], y_test, reference=lgb_train)

    gbm = lgb.train(default_params, lgb_train, num_boost_round=1000,
                    valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=50)
    y_pred = gbm.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The score is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))


def lightGBM_filter(train, test, params_best):
    print('LightGBM with filter method to do feature selection with best parameters')
    train, test = feature_select_pearson(train, test)
    # callback
    params_best = params_append(params_best)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    lgb_train = lgb.Dataset(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    lgb_eval = lgb.Dataset(x_test.loc[:, x_test.columns != 'card_id'], y_test, reference=lgb_train)

    gbm = lgb.train(params_best, lgb_train, num_boost_round=1000,
                    valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=50)
    y_pred = gbm.predict(x_test.loc[:, x_test.columns != 'card_id'])
    print('The score is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    result_folder = Path("../result/")
    result_folder.mkdir(parents=True, exist_ok=True)
    # add predict target to testing set
    x_test['predict_target'] = y_pred
    x_test[['card_id', 'predict_target']].to_csv(result_folder / 'lightgbm_filter.csv', index=False)


def light_GBM_wrapper(train, test, params_best):
    print('LightGBM with wrapper method to do feature selection with best parameters')
    train_ = train.copy()
    train, test = lightGBM_wrapper(train, test)

    data_folder = Path("../preprocess/")
    kaggle_test = pd.read_csv(data_folder / 'test.csv')
    _, kaggle_test = lightGBM_wrapper(train_, kaggle_test)
    # callback
    params_best = params_append(params_best)
    features = train.columns.tolist()
    features.remove("target")
    x_train = train[features]
    x_test = test[features]
    y_train = train['target']
    y_test = test['target']

    lgb_train = lgb.Dataset(x_train.loc[:, x_train.columns != 'card_id'], y_train)
    lgb_eval = lgb.Dataset(x_test.loc[:, x_test.columns != 'card_id'], y_test, reference=lgb_train)

    gbm = lgb.train(params_best, lgb_train, num_boost_round=1000,
                    valid_sets=lgb_eval, early_stopping_rounds=5)

    y_pred = gbm.predict(x_test.loc[:, x_test.columns != 'card_id'])

    print('The score is:')
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

    result_folder = Path("../result/")
    result_folder.mkdir(parents=True, exist_ok=True)
    # add predict target to testing set
    x_test['predict_target'] = y_pred
    x_test[['card_id', 'predict_target']].to_csv(result_folder / 'lightgbm_wrapper.csv', index=False)

    kaggle_test['target'] = gbm.predict(kaggle_test.loc[:, kaggle_test.columns != 'card_id'])
    kaggle_test[['card_id', 'target']].to_csv(result_folder / 'submission_lightGBM_wrapper.csv', index=False)


data_folder = Path("../preprocess/")
train = pd.read_csv(data_folder / 'new_train.csv')
test = pd.read_csv(data_folder / 'new_test.csv')

# user a small sample of training set to find the best parameters by gridsearch
train_sample = pd.read_csv(data_folder / 'new_train_30perc.csv')
# best_params = param_hyperopt(train_sample)

# Use the result of the best parameters we trained to save computing time
saved_best_params = {'learning_rate': 0.05548133558524738, 'reg_alpha': 5, 'reg_lambda': 5.897284630578619}

lightGBM_filter_default(train, test)
lightGBM_wrapper_default(train, test)
lightGBM_filter(train, test, saved_best_params)
light_GBM_wrapper(train, test, saved_best_params)
