import gc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def convert_object_type(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


# read datasets
data_folder = Path("elo-merchant-category-recommendation/")
train = pd.read_csv(data_folder / 'train.csv')
test = pd.read_csv(data_folder / 'test.csv')
# train = pd.read_csv('elo-merchant-category-recommendation/train.csv')
# test = pd.read_csv('elo-merchant-category-recommendation/test.csv')


# fill in the missing value with -1, and then convert all discrete fields to string types
merchant = pd.read_csv(data_folder / 'merchants.csv')
transaction = pd.read_csv(data_folder / 'new_merchant_transactions.csv')
# merchant = pd.read_csv('elo-merchant-category-recommendation/merchants.csv')
# transaction = pd.read_csv('elo-merchant-category-recommendation/new_merchant_transactions.csv')

# add a dir to store all preprocessed data
path = Path('preprocess')
path.mkdir(parents=True, exist_ok=True)

# encode the first active month
# se_map = convert_object_type(train['first_active_month'].append(test['first_active_month']).astype(str))
se_map = convert_object_type(pd.concat([train['first_active_month'], test['first_active_month']]).astype(str))
train['first_active_month'] = se_map[:train.shape[0]]
test['first_active_month'] = se_map[train.shape[0]:]

train.to_csv("preprocess/train_pre.csv", index=False)
test.to_csv("preprocess/test_pre.csv", index=False)

del train
del test
gc.collect()

# separates discrete features and contiguous features of merchant
category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                 'subsector_id', 'category_1',
                 'most_recent_sales_range', 'most_recent_purchases_range',
                 'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
                'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']

# dictionary sort encoding for non-numeric discrete fields
for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = convert_object_type(merchant[col])

# missing values of discrete features are filled with -1
merchant[category_cols] = merchant[category_cols].fillna(-1)

# the positive infinite value of discrete features is replaced by the maximum value
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())

# the average value is filled with the remaining null value
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())

# remove duplicates columns from the table
duplicate_cols = ['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id',
                  'category_2']
merchant = merchant.drop(duplicate_cols[1:], axis=1)
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)

# separates discrete features and contiguous features of translation
numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
                 'subsector_id']
time_cols = ['purchase_date']

# dictionary sort encoding for non-numeric discrete fields
for col in ['authorized_flag', 'category_1', 'category_3']:
    transaction[col] = convert_object_type(transaction[col].fillna(-1).astype(str))
transaction[category_cols] = transaction[category_cols].fillna(-1)
transaction['category_2'] = transaction['category_2'].astype(int)

# processing of time periods
transaction['purchase_month'] = transaction['purchase_date'].apply(lambda x: '-'.join(x.split(' ')[0].split('-')[:2]))
transaction['purchase_hour_section'] = transaction['purchase_date'].apply(
    lambda x: x.split(' ')[1].split(':')[0]).astype(int) // 6
transaction['purchase_day'] = transaction['purchase_date'].apply(
    lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday()) // 5
del transaction['purchase_date']

# dictionary sort encoding for purchase month
transaction['purchase_month'] = convert_object_type(transaction['purchase_month'].fillna(-1).astype(str))

# merge the tables
cols = ['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
transaction = pd.merge(transaction, merchant[cols], how='left', on='merchant_id')

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

transaction[cols[1:]] = transaction[cols[1:]].fillna(-1).astype(int)
transaction[category_cols] = transaction[category_cols].fillna(-1).astype(str)
transaction.to_csv(path / 'transaction_d_pre.csv', index=False)
del transaction
gc.collect()


# Add two new columns, purchase_day_diff and purchase_month_diff, which allows to calculate purchase_day/month by grouping by cardId
merchant = pd.read_csv(data_folder / 'merchants.csv')
transaction = pd.read_csv(data_folder / 'new_merchant_transactions.csv')

category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                 'subsector_id', 'category_1',
                 'most_recent_sales_range', 'most_recent_purchases_range',
                 'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
                'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']

for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = convert_object_type(merchant[col])

merchant[category_cols] = merchant[category_cols].fillna(-1)

inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())

for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())

duplicate_cols = ['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id',
                  'category_2']
merchant = merchant.drop(duplicate_cols[1:], axis=1)
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)

numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
                 'subsector_id']

for col in ['authorized_flag', 'category_1', 'category_3']:
    transaction[col] = convert_object_type(transaction[col].fillna(-1).astype(str))
transaction[category_cols] = transaction[category_cols].fillna(-1)
transaction['category_2'] = transaction['category_2'].astype(int)

transaction['purchase_month'] = transaction['purchase_date'].apply(lambda x: '-'.join(x.split(' ')[0].split('-')[:2]))
transaction['purchase_hour_section'] = transaction['purchase_date'].apply(
    lambda x: x.split(' ')[1].split(':')[0]).astype(int) // 6
transaction['purchase_day'] = transaction['purchase_date'].apply(
    lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday()) // 5
del transaction['purchase_date']

transaction['purchase_month'] = convert_object_type(transaction['purchase_month'].fillna(-1).astype(str))

cols = ['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
transaction = pd.merge(transaction, merchant[cols], how='left', on='merchant_id')

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

transaction['purchase_day_diff'] = transaction.groupby("card_id")['purchase_day'].diff()
transaction['purchase_month_diff'] = transaction.groupby("card_id")['purchase_month'].diff()
transaction.to_csv(path / 'transaction_g_pre.csv', index=False)
