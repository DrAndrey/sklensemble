# -*- coding: utf-8 -*-

"""

"""

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin


class BlendingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, is_need_weights=False):
        self.estimators = estimators
        self.is_need_weights = is_need_weights

        if not self.is_need_weights:
            self.is_need_weights = [False] * len(self.estimators)

        assert len(self.estimators) == len(self.is_need_weights)

    def fit(self, x, y, w=None):
        for estimator, is_need_weight in zip(self.estimators, self.is_need_weights):
            if is_need_weight:
                estimator.fit(x, y, w)
            else:
                estimator.fit(x, y)

    def predict(self, x):
        predictions = [estimator.predict(x) for estimator in self.estimators]
        df = pd.DataFrame(predictions).transpose()
        y = df.mode(axis=1)
        y.fillna(y.mode().iloc[0], inplace=True)
        y = y.iloc[:, 0]
        return y.values

    def predict_proba(self, x):
        predictions = [estimator.predict_proba(x) for estimator in self.estimators]
        panel = pd.Panel(predictions)
        proba = panel.apply(lambda el: el.mean(), axis=0)
        return proba.values


if __name__ == '__main__':
    pass
