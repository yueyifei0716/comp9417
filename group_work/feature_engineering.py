import gc
import pandas as pd
from pathlib import Path

# read the tables after preprocessing
data_folder = Path("preprocess/")
train = pd.read_csv(data_folder / 'train_pre.csv')
test = pd.read_csv(data_folder / 'test_pre.csv')
transaction = pd.read_csv(data_folder / 'transaction_d_pre.csv')

numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']


features = {}
# card_all = train['card_id'].append(test['card_id']).values.tolist()
card_all = pd.concat([train['card_id'], test['card_id']]).values.tolist()
for card in card_all:
    features[card] = {}

columns = transaction.columns.tolist()
idx = columns.index('card_id')
category_cols_index = [columns.index(col) for col in category_cols]
numeric_cols_index = [columns.index(col) for col in numeric_cols]

for i in range(transaction.shape[0]):
    va = transaction.loc[i].values
    card = va[idx]
    for cate_ind in category_cols_index:
        for num_ind in numeric_cols_index:
            col_name = '&'.join([columns[cate_ind], str(va[cate_ind]), columns[num_ind]])
            features[card][col_name] = features[card].get(col_name, 0) + va[num_ind]
del transaction
gc.collect()

# dict to dataframe
df = pd.DataFrame(features).T.reset_index()
del features
cols = df.columns.tolist()
df.columns = ['card_id'] + cols[1:]

# generate the training set and testing set
train = pd.merge(train, df, how='left', on='card_id')
test = pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv(data_folder / 'train_dict.csv', index=False)
test.to_csv(data_folder / 'test_dict.csv', index=False)

gc.collect()


transaction = pd.read_csv(data_folder / 'transaction_g_pre.csv')

numeric_cols = ['authorized_flag', 'category_1', 'installments',
                'category_3', 'month_lag', 'purchase_month', 'purchase_day', 'purchase_day_diff', 'purchase_month_diff',
                'purchase_amount', 'category_2',
                'purchase_month', 'purchase_hour_section', 'purchase_day',
                'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
categorical_cols = ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']


# calculate statistics values like mean, var...
store = {}
for col in numeric_cols:
    store[col] = ['nunique', 'mean', 'min', 'max', 'var', 'skew', 'sum']
for col in categorical_cols:
    store[col] = ['nunique']
store['card_id'] = ['size', 'count']
cols = ['card_id']

for key in store.keys():
    cols.extend([key + '_' + stat for stat in store[key]])

df = transaction[transaction['month_lag'] < 0].groupby('card_id').agg(store).reset_index()
df.columns = cols[:1] + [co + '_hist' for co in cols[1:]]

df2 = transaction[transaction['month_lag'] >= 0].groupby('card_id').agg(store).reset_index()
df2.columns = cols[:1] + [co + '_new' for co in cols[1:]]
df = pd.merge(df, df2, how='left', on='card_id')

df2 = transaction.groupby('card_id').agg(store).reset_index()
df2.columns = cols
df = pd.merge(df, df2, how='left', on='card_id')
del transaction
gc.collect()

# generate the training set and testing set
train = pd.merge(train, df, how='left', on='card_id')
test = pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv(data_folder / 'train_groupby.csv', index=False)
test.to_csv(data_folder / 'test_groupby.csv', index=False)

gc.collect()


train_dict = pd.read_csv(data_folder / 'train_dict.csv')
test_dict = pd.read_csv(data_folder / 'test_dict.csv')
train_groupby = pd.read_csv(data_folder / 'train_groupby.csv')
test_groupby = pd.read_csv(data_folder / 'test_groupby.csv')

for co in train_dict.columns:
    if co in train_groupby.columns and co != 'card_id':
        del train_groupby[co]
for co in test_dict.columns:
    if co in test_groupby.columns and co != 'card_id':
        del test_groupby[co]

train = pd.merge(train_dict, train_groupby, how='left', on='card_id').fillna(0)
test = pd.merge(test_dict, test_groupby, how='left', on='card_id').fillna(0)

train.to_csv(data_folder / 'train.csv', index=False)
test.to_csv(data_folder / 'test.csv', index=False)
