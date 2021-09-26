import logging
from time import time
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score

import optuna.integration.lightgbm as lgb
from optuna.integration.lightgbm import Dataset


class ClassifierLeaf:
    def __init__(self, cat_columns: list = None, value: int = 0):
        self.cat_columns = cat_columns
        self.value = value
        self.classifier = None
        self._val = None
        self._test = None

    def binarizer(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda x: 0 if x == self.value else 1)

    def _datasets(self, train: DataFrame, val: DataFrame) -> (Dataset, Dataset):
        x_train = train.drop(['ds', 'y'], axis=1)
        y_train = train.y.values
        features = list(x_train.columns)
        x_test = val[features].copy()
        y_test = val.y.values
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.cat_columns)
        dval = Dataset(x_test, label=y_test, categorical_feature=self.cat_columns)

        return dtrain, dval

    def _fit(self, train: DataFrame, val: DataFrame) -> None:
        dtrain, dval = self._datasets(train, val)

        params = {"objective": "binary",
                  "metric": "auc",
                  "verbosity": -1,
                  "boosting_type": "gbdt"}

        model = lgb.tuner.train(params, dtrain, valid_sets=[dtrain, dval], verbose_eval=0, early_stopping_rounds=200)

        self.classifier = model

    def _predict(self, test: DataFrame):
        return self.classifier.predict(test, num_iteration=self.classifier.best_iteration)

    def fit_predict(self, train: DataFrame, val: DataFrame, test: DataFrame) -> np.ndarray:
        logging.info(f'Classifier for value {self.value} started')
        start = time()

        x_train = train.copy()
        x_train['y'] = self.binarizer(x_train.y)
        x_val = val.copy()
        x_val['y'] = self.binarizer(x_val.y)
        x_test = test.copy()
        self._fit(x_train, x_val)
        self._val = self._predict(x_val.drop(['ds', 'y'], axis=1))
        self._test = self._predict(x_test)

        logging.info(f'Accuracy Validation: {accuracy_score(x_val.y.values, np.rint(self._val))}')
        logging.info(f'Classification made in {round(time() - start, 4)} sec')

        return np.rint(self._test)

    def get_leaf_data(self) -> (np.ndarray, np.ndarray):
        return self._val, self._test

    def set(self, val: np.ndarray, test: np.ndarray) -> None:
        self._val = val
        self._test = test
