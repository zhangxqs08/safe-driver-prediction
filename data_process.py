__author__ = 'Xiang'


import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

null_cols = df_train.columns[df_train[df_train == -1].any()]
to_drop = null_cols[[0, 1, 2, 3, 4, 6, 7, 8, 9, 11]]

train = df_train.drop(to_drop, axis=1)
test = df_test.drop(to_drop, axis=1)
y = train['target']
train = train.drop(['id', 'target'], axis=1)
test = test.drop('id', axis=1)
X = train.as_matrix().astype(np.float)
y = y.as_matrix().astype(np.float)
T = test.as_matrix().astype(np.float)

np.save('X_train.data', X)
np.save('y_train.data', y)
np.save('test.data', T)

