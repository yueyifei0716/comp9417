import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pylab as pl
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
pd.set_option('display.max_columns', None)


def feature_select_pearson(train):
    print('feature_select...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    featureSelect = features[:]

    for fea in features:
        if train[fea].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(fea)

    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = ['card_id'] + se[:300].index.tolist()
    print('done')
    return train[feature_select + ['target']]


data_folder = Path("../preprocess/")
origian_data = pd.read_csv(data_folder / 'new_train.csv')
col_num = origian_data.shape[0]
origian_data = feature_select_pearson(origian_data).drop('card_id', axis=1)

train = np.array(origian_data.iloc[:, :]).astype(float)
train_all = np.array(origian_data.iloc[:, :]).astype(float)
x_train = torch.tensor(train[:, :-1]).to(torch.float32)
y_train = torch.tensor(train[:, -1]).to(torch.float32)


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        x = x.squeeze(-1)
        return x


net = Net(300, 100, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_function = torch.nn.MSELoss()
loss_value = []
RSEM_loss = 0
for t in range(200):
    prediction = net(x_train)
    RSEM_loss = torch.sqrt(loss_function(prediction, y_train))
    loss_value.append(float(RSEM_loss))
    optimizer.zero_grad()
    RSEM_loss.backward()
    optimizer.step()
print(RSEM_loss)
fig = plt.figure(figsize=(7, 5))
pl.plot(range(200), loss_value, 'g-', label=u'Loss Value')
pl.legend()
plt.xlabel(u'iters')
plt.ylabel(u'loss value')
plt.title('loss curve')
pl.show()
