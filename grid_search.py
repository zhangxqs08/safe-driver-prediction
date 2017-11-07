__author__ = 'Xiang'


import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


X = np.load('X_train.data.npy')
y = np.load('y_train.data.npy')
T = np.load('test.data.npy')

n_splits = 5

# define evaluation metrics
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert(len(actual) == len(pred))
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1*all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


def grid_search(clc, clc_para, parameters):
    gs = GridSearchCV(clc(**clc_para), parameters, scoring='roc_auc', n_jobs=4, cv=5, verbose=2)
    gs.fit(X, y)
    print "best parameters:", gs.best_params_
    print "best score:", gs.best_score_


if __name__ == '__main__':
    model = XGBClassifier
    model_para = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'seed': 27
    }
    parameters = {'max_depth': [5, 9]}
    grid_search(model, model_para, parameters)

