import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb


def feature_select_pearson(train, test):
    print('Filter method to select top 300 features...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    featureSelect = features[:]

    # remove features with a missing value ratio greater than 0.99
    for feature in features:
        if train[feature].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(feature)

    # calculate the pearson correlation
    corr = []
    for feature in featureSelect:
        corr.append(abs(train[[feature, 'target']].fillna(0).corr().values[0][1]))

    # get top 300 features with the highest correlation
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = ['card_id'] + se[:300].index.tolist()
    print('done')
    return train[feature_select + ['target']], test[feature_select + ['target']]


def xgboost_wrapper(train, test):
    print('Wrapper method to select top 300 features...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # the best parameters we get from grid search with cross validation
    params_initial = {
        'max_depth': 6,
        'n_estimators': 1000,
        'learning_rate': 0.2,
        'objective': "reg:squarederror",
        'seed': 42
    }
    # cross validation
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        clf = xgb.XGBRegressor(**params_initial)
        clf.fit(train[features].loc[train_part_index], train[label].loc[train_part_index], early_stopping_rounds=5,
                eval_set=[(train[features].loc[eval_index], train[label].loc[eval_index])])
        fse += pd.Series(clf.feature_importances_, features)

    # get top 300 features with the highest correlation
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    test_features = test.columns.tolist()
    if 'target' in test_features:
        return train[feature_select + ['target']], test[feature_select + ['target']]
    else:
        return train[feature_select + ['target']], test[feature_select]


def random_forest_wrapper(train, test):
    print('Wrapper method to select top 300 features...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # the best parameters we get from grid search with cross validation
    params_initial = {
        "n_estimators": 80,
        "min_samples_leaf": 31,
        "min_samples_split": 2,
        "max_depth": 10,
        "max_features": 80,
        'criterion': "squared_error",
        'n_jobs': 15,
        'random_state': 42
    }
    # cross validation
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        clf = RandomForestRegressor(**params_initial)
        clf.fit(train[features].loc[train_part_index], train[label].loc[train_part_index])
        fse += pd.Series(clf.feature_importances_, features)

    # get top 300 features with the highest correlation
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    test_features = test.columns.tolist()
    if 'target' in test_features:
        return train[feature_select + ['target']], test[feature_select + ['target']]
    else:
        return train[feature_select + ['target']], test[feature_select]


def lightGBM_wrapper(train, test):
    print('Wrapper method to select top 300 features...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # the best parameters we get from grid search with cross validation
    params_initial = {
        'learning_rate': 0.03348897652973219,
        'reg_alpha': 6,
        'reg_lambda': 7.392182380711307,
        'objective': 'regression',
        'metric': 'rmse',
        'feature_pre_filter': False,
        'bagging_seed': 42
    }

    # cross validation
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        bst = lgb.train(params_initial, train_part, num_boost_round=1000,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=30, verbose_eval=50)
        fse += pd.Series(bst.feature_importance(), features)

    # get top 300 features with the highest correlation
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    test_features = test.columns.tolist()
    if 'target' in test_features:
        return train[feature_select + ['target']], test[feature_select + ['target']]
    else:
        return train[feature_select + ['target']], test[feature_select]
