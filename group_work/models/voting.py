import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path


# read the results from the best three models
result_folder = Path("../result/")
data_folder = Path("../preprocess/")

data = pd.read_csv(result_folder / 'randomforest_wrapper.csv')
data['randomforest'] = data['predict_target'].values

temp = pd.read_csv(result_folder / 'lightgbm_wrapper.csv')
data['lightgbm'] = temp['predict_target'].values

temp = pd.read_csv(result_folder / 'xgboost_wrapper.csv')
data['xgboost'] = temp['predict_target'].values

# compute pairwise correlation of columns
print(data.corr())

y_test = pd.read_csv(data_folder / 'new_test.csv')['target']

data['voting_target_avg'] = (data['randomforest'] + data['lightgbm'] + data['xgboost']) / 3
data[['card_id', 'voting_target_avg']].to_csv(result_folder / 'voting_avg.csv', index=False)
y_pred = data['voting_target_avg']

print('The RMSE is:')
print(np.sqrt(mean_squared_error(y_test, y_pred)))

data['voting_target_weight'] = data['randomforest'] * 0.25 + data['lightgbm'] * 0.5 + data['xgboost'] * 0.25
data[['card_id', 'voting_target_weight']].to_csv(result_folder / 'voting_weight.csv', index=False)
y_pred = data['voting_target_weight']

print('The RMSE is:')
print(np.sqrt(mean_squared_error(y_test, y_pred)))


# for kaggle testing set
data = pd.read_csv(result_folder / 'submission_randomforest_wrapper.csv')
data['randomforest'] = data['target'].values

temp = pd.read_csv(result_folder / 'submission_lightGBM_wrapper.csv')
data['lightgbm'] = temp['target'].values

temp = pd.read_csv(result_folder / 'submission_xgboost_wrapper.csv')
data['xgboost'] = temp['target'].values

print(data.corr())

data['target'] = (data['randomforest'] + data['lightgbm'] + data['xgboost']) / 3
data[['card_id', 'target']].to_csv(result_folder / 'submission_voting_avg.csv', index=False)


data['target'] = data['randomforest'] * 0.25 + data['lightgbm'] * 0.5 + data['xgboost'] * 0.25
data[['card_id', 'target']].to_csv(result_folder / 'submission_voting_weight.csv', index=False)

