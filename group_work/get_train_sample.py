import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


data_folder = Path("preprocess/")
origian_data = pd.read_csv(data_folder / 'new_train.csv')
features = origian_data.drop(['card_id', 'target'], axis=1).columns
col_num = origian_data.shape[0]
train = origian_data.iloc[:int(col_num * 0.3), :]
train_all = origian_data.iloc[:, :]
train_num = train.shape[0]
train_all_num = train_all.shape[0]


def unit_value_rule(features):
    for feature in features:
        (train[feature].value_counts().sort_index() / train_num).plot()
        (train_all[feature].value_counts().sort_index() / train_all_num).plot()
        plt.legend(['train', 'train_all'])
        plt.xlabel(feature)
        plt.ylabel('ratio')
        plt.show()


def value_combine(feature_1, feature_2, df):
    feature1 = df[feature_1].astype(str).values.tolist()
    feature2 = df[feature_2].astype(str).values.tolist()
    return pd.Series([feature1[i] + '&' + feature2[i] for i in range(df.shape[0])])


def multiple_value_rule(features):
    for feature in features[1:]:
        train_value = (value_combine(features[0], feature, train).value_counts().sort_index()) / train_num
        train_all_value = (value_combine(features[0], feature, train_all).value_counts().sort_index()) / train_all_num
        index_value = pd.Series(
            train_value.index.tolist() + train_all_value.index.tolist()).drop_duplicates().sort_values()
        (index_value.map(train_value).fillna(0)).plot()
        (index_value.map(train_all_value).fillna(0)).plot()
        plt.legend(['train', 'train_all'])
        plt.xlabel('&'.join([features[0], feature]))
        plt.ylabel('ratio')
        plt.show()


# you can comment out this to run the functions above and see how the 30% of the training
# set has the similar distribution with the whole training set

# unit_value_rule(features)
# month = train['first_active_month'].tolist()
# month_dic = {}
# for i in month:
#     if i in month_dic.keys():
#         month_dic[i] += 1
#     else:
#         month_dic[i] = 1
#
# for num in month_dic.keys():
#     month_dic[num] = month_dic[num] / col_num
# month_dic = pd.DataFrame(list(month_dic.items()), columns=['Month', 'Ratio']).sort_values(by='Month')
# print(month_dic)
# for i in range(0, month_dic.shape[0]):
#     print(month_dic['Month'][i], '{:.4%}'.format(month_dic["Ratio"][i]))

train.to_csv(data_folder / 'new_train_30perc.csv', index=False)
