import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


data_folder = Path("preprocess/")
data = pd.read_csv(data_folder / 'train.csv')

features = data.columns.tolist()
features.remove('target')
x_data = data[features]
y_data = data['target']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

train = pd.concat([x_train, y_train], axis=1)
test = pd.concat([x_test, y_test], axis=1)

print('Splitting the given training set to new training set and testing set')
train.to_csv(data_folder / 'new_train.csv', index=False)
test.to_csv(data_folder / 'new_test.csv', index=False)
print('Splitting done')
