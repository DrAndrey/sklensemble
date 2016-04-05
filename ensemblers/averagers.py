# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize

from sklearn.base import BaseEstimator, ClassifierMixin


class BlendingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, is_need_weights=False, x_mask=None):
        self.estimators = estimators
        self.is_need_weights = is_need_weights
        self.x_mask = x_mask

        if not self.is_need_weights:
            self.is_need_weights = [False] * len(self.estimators)

        if not self.x_mask:
            self.x_mask = [None] * len(self.estimators)

        assert len(self.estimators) == len(self.is_need_weights)
        assert len(self.estimators) == len(self.x_mask)

    def fit(self, x, y, w=None):
        for estimator, is_need_weight, x_mask in zip(self.estimators, self.is_need_weights, self.x_mask):
            print("{0} is fitting".format(estimator))

            if x_mask is None:
                x_with_mask = x
            else:
                assert len(x_mask) == x.shape[1]
                x_with_mask = x[:, x_mask]

            if is_need_weight:
                estimator.fit(x_with_mask, y, w)
            else:
                estimator.fit(x_with_mask, y)

    def predict(self, x):
        predictions = [estimator.predict(x) if x_mask is None else estimator.predict(x[:, x_mask])
                       for estimator, x_mask in zip(self.estimators, self.x_mask)]
        df = pd.DataFrame(predictions).transpose()
        y = df.mode(axis=1)
        y.fillna(y.mode().iloc[0], inplace=True)
        y = y.iloc[:, 0]
        return y.values

    def predict_proba(self, x):
        predictions = [estimator.predict_proba(x) if x_mask is None else estimator.predict_proba(x[:, x_mask])
                       for estimator, x_mask in zip(self.estimators, self.x_mask)]
        panel = pd.Panel(predictions)
        return panel.values

    def predict_optimized_proba(self, x_train, y_train, x_test):
        classes = np.unique(y_train)
        assert (classes == np.array(range(len(classes)))).all()

        n_est = len(self.estimators)
        train_proba = self.predict_proba(x_train)

        train_pred_class_proba = []
        for sample_ind, sample_class in enumerate(y_train):
            train_pred_class_proba.append(train_proba[:, sample_ind, sample_class])
        train_pred_class_proba = np.array(train_pred_class_proba).mean() / n_est

        opt_fun = lambda x: (1 - (np.array(x) * train_pred_class_proba).sum()) ** 2
        first_w = np.ones(n_est) / n_est

        opt_w = optimize.minimize(opt_fun, tuple(first_w)).x
        print("simple_mean_acc - {0}, opt_weighted_mean_acc - {1}, opt_weights - {2}".format(opt_fun(first_w),
                                                                                             opt_fun(opt_w), opt_w))
        y_test = self.predict(x_test)
        test_proba = self.predict_proba(x_test)
        test_pred_class_proba = []
        for sample_ind, sample_class in enumerate(y_test):
            test_pred_class_proba.append((test_proba[:, sample_ind, sample_class]*opt_w).sum())
        return test_pred_class_proba

if __name__ == '__main__':
    pass
